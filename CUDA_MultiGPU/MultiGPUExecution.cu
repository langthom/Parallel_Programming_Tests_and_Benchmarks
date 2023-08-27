/*
 * MIT License
 *
 * Copyright (c) 2023 Dr. Thomas Lang
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
**/

#include <algorithm>
#include <cassert>
#include <numeric>
#include "MultiGPUExecution.h"
#include "../Common/CommonKernels.h"

#ifdef HAS_CUDA

#include "../Common/StatisticsKernel.h"


double getMaxAllocationSizeMultiCUDAGPU(double maxMemoryInGiB, double actuallyUsePercentage) {
  constexpr size_t toGiB = 1ull << 30;
  size_t maxMemoryInBytes = static_cast< size_t >(maxMemoryInGiB * toGiB);

  // Detect all available CUDA devices.
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  if (error == cudaError_t::cudaSuccess) {
    // Compute the total size available on all devices.
    size_t totalAvailableOnAllDevices = std::accumulate(availableMemoryPerDevice.cbegin(), availableMemoryPerDevice.cend(), 0ull);
    maxMemoryInBytes = std::min< size_t >(maxMemoryInBytes, totalAvailableOnAllDevices);
  }

  double maxMemoryInGB = static_cast< double >(maxMemoryInBytes) / toGiB;
  maxMemoryInGB *= actuallyUsePercentage;
  return maxMemoryInGB;
}


cudaError_t launchKernelMultiCUDAGPU(float* out, float* in, size_t N, int* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTime, int deviceID, int threadsPerBlock) {
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  assert(out); assert(in); assert(offsets); assert(elapsedTime);

  cudaError_t error = cudaError_t::cudaSuccess;
  int nr_gpus = 0;
  std::vector< double > memoryFractionsPerDevice(nr_gpus);
  {
    std::vector< std::string > deviceNames;
    std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
    error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);
    HANDLE_ERROR(error);
    size_t totalMemoryAvailable = std::accumulate(availableMemoryPerDevice.cbegin(), availableMemoryPerDevice.cend(), 0ull);

    std::transform(availableMemoryPerDevice.cbegin(), availableMemoryPerDevice.cend(), memoryFractionsPerDevice.begin(),
                  [totalMemoryAvailable](size_t availablePerDevice) { return static_cast< double >(availablePerDevice) / totalMemoryAvailable; });
  }

  // 1. Create a stream for each device.
  std::vector< cudaStream_t > streams(nr_gpus);
  for (cudaStream_t& stream : streams) {
    error = cudaStreamCreate(&stream);
  }
  HANDLE_ERROR(error);


  // 2. Memory allocation per device.
  size_t offsetBytes      = K * K * K * sizeof(int);
  size_t totalVolumeBytes = N * sizeof(float);
  size_t sliceSize        = dimX * dimY;

  std::vector< size_t > gpuAllocationSizes(nr_gpus);
  std::vector< float* > gpuAllocations(nr_gpus * 3);
  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    cudaStream_t& currentStream = streams[gpuID];

    // allocation size which is always a multiple of slices.
    size_t allocationForCurrentDevice = static_cast< size_t >(totalVolumeBytes * memoryFractionsPerDevice[gpuID]);
    allocationForCurrentDevice -= allocationForCurrentDevice % sliceSize;
    if (gpuID + 1 == nr_gpus) {
      allocationForCurrentDevice = totalVolumeBytes - std::accumulate(gpuAllocationSizes.cbegin(), gpuAllocationSizes.cend() - 1/*excluding the last, this is computed here*/, 0ull);
    }
    gpuAllocationSizes[gpuID] = allocationForCurrentDevice;

    // Do the actual allocation
    error = cudaSetDevice(gpuID);
    HANDLE_ERROR(error);
    error = cudaMallocAsync(reinterpret_cast< void** >(&gpuAllocations[2*gpuID+0]), allocationForCurrentDevice, currentStream); // The input chunk.
    HANDLE_ERROR(error);
    error = cudaMallocAsync(reinterpret_cast< void** >(&gpuAllocations[2*gpuID+1]), allocationForCurrentDevice, currentStream); // The output chunk.
    HANDLE_ERROR(error);
    error = cudaMallocAsync(reinterpret_cast< void** >(&gpuAllocations[2*gpuID+2]), offsetBytes,                currentStream); // The offsets for jumping.
    HANDLE_ERROR_STMT(error, cudaFree(gpuAllocations[gpuID]));
  }

  auto freeAllocations = [&gpuAllocations](){
    // TODO: sync?
    for (auto& allocPtr : gpuAllocations) {
      cudaFree(allocPtr);
    }
  };

  std::exit(1); // DELME

  //

  // TODO: global timer

  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    error = cudaSetDevice(gpuID);
    HANDLE_ERROR_STMT(error, freeAllocations());

    cudaStream_t& currentStream = streams[gpuID];
    float* deviceInp = gpuAllocations[2 * gpuID + 0];
    float* deviceOut = gpuAllocations[2 * gpuID + 1];
    float* deviceOff = gpuAllocations[2 * gpuID + 2];

    // Copy the allocation asynchronously to the devices.
    size_t startOfCurrentChunk = std::accumulate(gpuAllocationSizes.cbegin(), gpuAllocationSizes.cbegin() + gpuID, 0ull); // TODO: Halos
    error = cudaMemcpyAsync(deviceInp, in + startOfCurrentChunk, gpuAllocationSizes[gpuID], cudaMemcpyHostToDevice, currentStream);
    error = cudaMemcpyAsync(deviceOff, offsets,                  offsetBytes,               cudaMemcpyHostToDevice, currentStream);
    HANDLE_ERROR_STMT(error, freeAllocations());


    // TODO: check?
    size_t voxelsForCurrentDeviceChunk = gpuAllocationSizes[gpuID] / sizeof(float);
    int blocksPerGrid = (voxelsForCurrentDeviceChunk + threadsPerBlock - 1) / threadsPerBlock;

    statisticsKernel<<< blocksPerGrid, threadsPerBlock, 0, currentStream >>>(deviceInp, deviceOut, deviceOff, K, N, dimX, dimY, dimZ);

    // TODO: halos?
    error = cudaMemcpyAsync(out + startOfCurrentChunk, deviceOut, gpuAllocationSizes[gpuID], cudaMemcpyDeviceToHost, currentStream);
    HANDLE_ERROR_STMT(error, freeAllocations());
  }

  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    error = cudaSetDevice(gpuID);
    HANDLE_ERROR(error);

    cudaStreamSynchronize(streams[gpuID]);
  }

  // TODO: timings, write back into parameter


  freeAllocations();

  for (auto& stream : streams) {
    error = cudaStreamDestroy(stream);
    HANDLE_ERROR(error);
  }

  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}

#endif // HAS_CUDA
