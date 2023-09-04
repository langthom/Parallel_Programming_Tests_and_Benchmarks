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

#include "../Common/CommonFunctions.h" // DELME
#include <iostream> // DELME

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


std::vector< std::pair< std::size_t, std::size_t > > partitionVolumeForMultiGPU(int& nr_gpus, cudaError_t& error, int K, std::size_t dimZ) {
  std::vector< double > memoryFractionsPerDevice;
  {
    std::vector< std::string > deviceNames;
    std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
    error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);
    if(error != cudaError_t::cudaSuccess) {
      return {};
    }

    size_t totalMemoryAvailable = std::accumulate(availableMemoryPerDevice.cbegin(), availableMemoryPerDevice.cend(), 0ull);

    memoryFractionsPerDevice.resize(nr_gpus);
    std::transform(availableMemoryPerDevice.cbegin(), availableMemoryPerDevice.cend(), memoryFractionsPerDevice.begin(),
                  [totalMemoryAvailable](size_t availablePerDevice) { return static_cast< double >(availablePerDevice) / totalMemoryAvailable; });
  }

  std::vector< std::size_t > chunkZs(nr_gpus);
  chunkZs[0] = 0;
  for(int gpuID = 1; gpuID < nr_gpus; ++gpuID) {
    chunkZs[gpuID] = chunkZs[gpuID-1] + static_cast< std::size_t >(static_cast< double >(dimZ) * memoryFractionsPerDevice[gpuID-1]);
  }


  std::vector< std::pair< std::size_t, std::size_t > > partitions;
  partitions.reserve(nr_gpus);
  std::size_t K2 = K / 2ull;

  for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    std::size_t begin = gpuID     == 0       ? 0ull     : chunkZs[gpuID] - K2;
    std::size_t end   = gpuID + 1 == nr_gpus ? dimZ - 1 : chunkZs[gpuID + 1];
    partitions.emplace_back(begin, end);
  }

  return partitions;
}



cudaError_t launchKernelMultiCUDAGPU(float* out, float* in, size_t N, int* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTime, int threadsPerBlock) {
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  assert(out); assert(in); assert(offsets); assert(elapsedTime);

  size_t totalDeviceMem = 0; // DELME

  cudaError_t error = cudaError_t::cudaSuccess;
  int nr_gpus = 0;

  // 1. Compute the partitioning over the devices, i.e., how many slices of the given block are processed
  //    on which device. The computed indices are z-slice indices which are inclusive on both sides.
  auto partitioning = partitionVolumeForMultiGPU(nr_gpus, error, K, dimZ);
  HANDLE_ERROR(error);

  // 2. Create a stream for each device.
  std::vector< cudaStream_t > streams(nr_gpus);
  for (cudaStream_t& stream : streams) {
    error = cudaStreamCreate(&stream);
  }
  HANDLE_ERROR(error);


  // 3. Memory allocation on the devices.
  //    Note that on older devices, the asynchronous API (i.e., cudaMallocAsync and its similar functions)
  //    may not be supported. Thus, we rely on the "older" way of doing things, namely doing the *blocking*
  //    calls, but concurrently on the CPU via regular CPU threads.

  std::vector< std::size_t > gpuDataAllocationSizes(nr_gpus * 2);
  std::vector< float* > gpuDataAllocations(nr_gpus * 2);
  std::vector< int* >   gpuOffsetAllocations(nr_gpus);

  auto freeAllocations = [&gpuDataAllocations,&gpuOffsetAllocations](){
    // TODO: sync?
    for (auto& allocPtr : gpuDataAllocations) {
      cudaFree(allocPtr);
    }
    for (auto& allocPtr : gpuOffsetAllocations) {
      cudaFree(allocPtr);
    }
  };

  std::size_t offsetBytes = K * K * K * sizeof(int);
  std::size_t sliceSizeBytes = dimX * dimY * sizeof(float);
  std::size_t padding = K / 2ull;

  #pragma omp parallel num_threads(nr_gpus)
  {
    cudaError_t threadLocal_error = cudaError_t::cudaSuccess;

    #pragma omp parallel for
    for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      // Change the current device.
      threadLocal_error = cudaSetDevice(gpuID);

      // Allocate memory for the data chunks (input and output) on the device.
      auto const& zPartition = partitioning[gpuID];
      std::size_t zRange = zPartition.second - zPartition.first + 1ull; /* +1 since the upper z boundary is inclusive */
      gpuDataAllocationSizes[2*gpuID+0] = zRange * sliceSizeBytes;
      gpuDataAllocationSizes[2*gpuID+1] = (zRange - 2*padding) * sliceSizeBytes;

      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuDataAllocations[2*gpuID+0]), gpuDataAllocationSizes[2*gpuID+0]);
      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuDataAllocations[2*gpuID+1]), gpuDataAllocationSizes[2*gpuID+1]);
      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuOffsetAllocations[gpuID]),   offsetBytes);

      #pragma omp critical
      if(threadLocal_error != cudaError_t::cudaSuccess) {
        error = threadLocal_error;
      }
    }
  }

  // explicit (CPU) synchronization for error handling, breaking from an openmp loop is ... weird
  HANDLE_ERROR_STMT(error, freeAllocations());

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  cudaEventRecord(startEvent);

  for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    cudaStream_t& currentStream = streams[gpuID];

    error = cudaSetDevice(gpuID);
    HANDLE_ERROR_STMT(error, freeAllocations());

    float* deviceDataIn  = gpuDataAllocations[2 * gpuID + 0];
    float* deviceDataOut = gpuDataAllocations[2 * gpuID + 1];
    int*   deviceOffsets = gpuOffsetAllocations[gpuID];

    std::size_t inputBytes  = gpuDataAllocationSizes[2 * gpuID + 0];
    std::size_t outputBytes = gpuDataAllocationSizes[2 * gpuID + 1];

    auto const& zPartition = partitioning[gpuID];
    float* dataInBegin  =  in + zPartition.first * sliceSizeBytes;
    float* dataOutBegin = out + (zPartition.first + padding) * sliceSizeBytes;

    error = cudaMemcpyAsync(deviceDataIn,  dataInBegin, inputBytes,  cudaMemcpyHostToDevice, currentStream);
    error = cudaMemcpyAsync(deviceOffsets, offsets,     offsetBytes, cudaMemcpyHostToDevice, currentStream);
    HANDLE_ERROR_STMT(error, freeAllocations(); std::cout << "async fwd-copy error at " << __LINE__ << "\n");

    int blocksPerGrid = (inputBytes / sizeof(float) + threadsPerBlock - 1) / threadsPerBlock;
    statisticsKernel<<< blocksPerGrid, threadsPerBlock, 0, currentStream >>>(deviceDataIn, deviceDataOut, deviceOffsets, K, N, dimX, dimY, dimZ);

    error = cudaMemcpy(dataOutBegin, deviceDataOut, outputBytes, cudaMemcpyDeviceToHost);
    //error = cudaMemcpyAsync(dataOutBegin, deviceOutBeg, dataOutCopySize, cudaMemcpyDeviceToHost, currentStream);
    HANDLE_ERROR_STMT(error, freeAllocations(); std::cout << "async back-copy error at " << __LINE__ << "\n");
  }

  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(elapsedTime, startEvent, stopEvent);

  // TODO: necessary?
  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    error = cudaSetDevice(gpuID);
    error = cudaStreamSynchronize(streams[gpuID]);
    HANDLE_ERROR_STMT(error, freeAllocations());
  }



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
