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
#include "CommonKernels.h"

#ifdef HAS_CUDA

#include "StatisticsKernel.h"

double getMaxAllocationSizeMultiCUDAGPU(double maxMemoryInGiB, double actuallyUsePercentage) {
  constexpr size_t toGiB = 1ull << 30;
  size_t maxMemoryInBytes = static_cast< size_t >(maxMemoryInGiB * toGiB);

  // Detect all available CUDA devices.
  int nr_gpus;
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
  // 1. Compute the fractions of available memory compared to the overall available memory.
  //    For example, if we have two identical devices which are not used currently, the fractions will likely be [0.5, 0.5].
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

  // 2. Compute the number of "z" slices assigned to each device, according to the fraction above.
  //    This way, all devices are under an approximately equal load (considered in percent).
  std::vector< std::size_t > chunkZs(nr_gpus);
  chunkZs[0] = 0;
  for(int gpuID = 1; gpuID < nr_gpus; ++gpuID) {
    chunkZs[gpuID] = chunkZs[gpuID-1] + static_cast< std::size_t >(static_cast< double >(dimZ) * memoryFractionsPerDevice[gpuID-1]);
  }

  // 3. Compute the partitions, i.e., the indices of the first and last slices to be assigned on a 
  //    single device. This also respects the necessary padding in between two successive z chunks.
  //    All indices are *inclusive*.
  std::vector< std::pair< std::size_t, std::size_t > > partitions;
  partitions.reserve(nr_gpus);
  std::size_t K2 = K / 2ull;

  for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    std::size_t begin = gpuID     == 0       ? 0ull     : chunkZs[gpuID]     - K2;
    std::size_t end   = gpuID + 1 == nr_gpus ? dimZ - 1 : chunkZs[gpuID + 1] + K2 - 1;
    partitions.emplace_back(begin, end);
  }

  return partitions;
}

cudaError_t launchKernelMultiCUDAGPU(float* out, float* in, int64_t N, int64_t* offsets, int K, int64_t dimX, int64_t dimY, int64_t dimZ, float* elapsedTime, int threadsPerBlock) {
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  assert(out); assert(in); assert(offsets); assert(elapsedTime);

  cudaError_t error = cudaError_t::cudaSuccess;
  int nr_gpus = 0;

  // 1. Compute the partitioning over the devices, i.e., how many slices of the given block are processed
  //    on which device. The computed indices are z-slice indices which are inclusive on both sides.
  auto partitioning = partitionVolumeForMultiGPU(nr_gpus, error, K, dimZ);
  HANDLE_ERROR(error);

  // 2. Prepare some data structures for the multi GPU processing:
  //     o timingEvents
  //         Events for tracking the beginning and the end of a CUDA execution (memory transfer and kernel execution)
  //         for each device.
  //     o gpuDataAllocationSizes
  //         Number of bytes of the input and output allocation for each device.
  //     o gpuDataAllocations
  //         Device allocation buffers for both the input and output data of the sizes specified in gpuDataAllocationSizes.
  //     o gpuOffsetAllocations
  //         Device allocations for offsets.
  //
  std::vector< cudaEvent_t >  timingEvents(2 * nr_gpus);
  std::vector< int64_t >      gpuDataAllocationSizes(nr_gpus * 2);
  std::vector< float* >       gpuDataAllocations(nr_gpus * 2);
  std::vector< int64_t* >     gpuOffsetAllocations(nr_gpus);

  auto freeAllocations = [nr_gpus,&gpuDataAllocations,&gpuOffsetAllocations](bool sync = true) {
    if(sync) {
      for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
        cudaSetDevice(gpuID);
        cudaDeviceSynchronize();
      }
    }
    for (auto& allocPtr : gpuDataAllocations) {
      cudaFree(allocPtr);
    }
    for (auto& allocPtr : gpuOffsetAllocations) {
      cudaFree(allocPtr);
    }
  };

  int padding                     = K - 1;
  int64_t offsetBytes             = K * K * K * sizeof(int64_t);
  int64_t sliceSize               = dimX * dimY;
  int64_t sliceSizeWithoutPadding = (dimX - padding) * (dimY - padding);

  float maxElapsedTime = 0.f;

  #pragma omp parallel num_threads(nr_gpus)
  {
    cudaError_t threadLocal_error = cudaError_t::cudaSuccess;

    #pragma omp for
    for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      // Change the current device.
      threadLocal_error = cudaSetDevice(gpuID);

      // 3. Create a stream for each device and events for synchronization/timing.
      //    Here we use the "cudaEventBlockingSync" flag to make the synchronization per device with this flag.
      threadLocal_error = cudaSetDevice(gpuID);
      threadLocal_error = cudaEventCreateWithFlags(&timingEvents[2*gpuID+0], cudaEventBlockingSync);
      threadLocal_error = cudaEventCreateWithFlags(&timingEvents[2*gpuID+1], cudaEventBlockingSync);

      int inputIndex  = 2 * gpuID + 0;
      int outputIndex = 2 * gpuID + 1;

      // Allocate memory for the data chunks (input and output) on the device.
      //  - The input memory blob includes the padding, as it was calculated by the partitioning function.
      //  - The output memory blob does not include the padding, thus we omit it.
      // Note that these are _blocking_ calls, yet executed in different threads for each device.
      auto const& zPartition = partitioning[gpuID];
      int64_t zRange = zPartition.second - zPartition.first + 1ull; /* +1 since the upper z boundary is inclusive */
      gpuDataAllocationSizes[ inputIndex] = zRange * sliceSize * sizeof(float);
      gpuDataAllocationSizes[outputIndex] = (zRange - padding) * sliceSizeWithoutPadding * sizeof(float);

      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuDataAllocations[ inputIndex]), gpuDataAllocationSizes[ inputIndex]);
      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuDataAllocations[outputIndex]), gpuDataAllocationSizes[outputIndex]);
      threadLocal_error = cudaMalloc(reinterpret_cast< void** >(&gpuOffsetAllocations[gpuID]), offsetBytes);
  
      // 4. Invoke the statistics kernel on each device by following the given pattern:
      //         o  Copy the input values (including padding) from the host to the device.
      //         o  Execute the kernel (which omits any padding directly).
      //         o  Copy the output values (without padding) back to the according host buffer.
      // 
      // Some notes on this:
      //  - Each kernel invocation gets a defined chunk with the necessary padding, and also its own
      //    execution grid. Thus, the kernels omit the full three-dimensional boundary and the index
      //    calculations within each kernel stay correct, regardless of using one or more GPUs.
      //  - The output buffer is assumed to be just as big as all computed values, i.e., the size of
      //    the input volume without padding. Consequently, only as much data points are written back
      //    to the buffer and the according statements do so by simple pointer arithmetic.

      // Capture the execution start.
      threadLocal_error = cudaEventRecord(timingEvents[inputIndex]);

      // Do the pointer arithmetic to calculate where the input slices (including padding) start,
      // which is given by the offset calculated in the partitioning function, and where the begin
      // of the output pointer is, i.e., where the non-padded computed results should be written to.
      // By construction of the partitioning function, the z-offset is the same, only the slice size
      // changes accordingly since no padding is present anymore.
      float*   deviceDataIn  = gpuDataAllocations[ inputIndex];
      float*   deviceDataOut = gpuDataAllocations[outputIndex];
      int64_t* deviceOffsets = gpuOffsetAllocations[gpuID];

      int64_t inputBytes  = gpuDataAllocationSizes[ inputIndex];
      int64_t outputBytes = gpuDataAllocationSizes[outputIndex];

      int partitionDimZ   = static_cast< int >(zPartition.second - zPartition.first + 1ull);
      float* dataInBegin  =  in + zPartition.first * sliceSize;
      float* dataOutBegin = out + zPartition.first * sliceSizeWithoutPadding;

      // Copy the input data from the host to the device.
      threadLocal_error = cudaMemcpy(deviceDataIn,  dataInBegin, inputBytes,  cudaMemcpyHostToDevice);
      threadLocal_error = cudaMemcpy(deviceOffsets, offsets,     offsetBytes, cudaMemcpyHostToDevice);

      // Invoke the kernel (asynchronously), which processes its own chunk of data of a smaller size (in general).
      // Each kernel will run in its own non-default stream, allowing for a true GPU multithreading approach.
      int64_t voxelsInCurrentPartition = partitionDimZ * sliceSize;
      int64_t blocksPerGrid = (voxelsInCurrentPartition + threadsPerBlock - 1) / threadsPerBlock;
      statisticsKernel<<< blocksPerGrid, threadsPerBlock >>>(deviceDataIn, deviceDataOut, deviceOffsets, K, voxelsInCurrentPartition, dimX, dimY, partitionDimZ);
      
      // Copy the computed result values back to the according buffer (asynchronously).
      threadLocal_error = cudaMemcpy(dataOutBegin, deviceDataOut, outputBytes, cudaMemcpyDeviceToHost);

      // 5. Compute the elapsed time. Since time measurements in any multithreaded environment are particularly
      //    difficult, we record the elapsed time of each individual device and take the maximum of it.
      //    We use the maximum of those values since they all enforce a synchronization and the total execution on
      //    all GPUs will take as long as the longest execution on any of its devices.
      //    Also, destroy the streams as they are no longer needed.

      float elapsedMs = 0.f;
      cudaEvent_t& stopEvent = timingEvents[2*gpuID+1];
      threadLocal_error = cudaEventRecord(stopEvent);
      threadLocal_error = cudaEventSynchronize(stopEvent);
      threadLocal_error = cudaEventElapsedTime(&elapsedMs, timingEvents[inputIndex], stopEvent);

      #pragma omp critical
      {
        maxElapsedTime = std::max< float >(maxElapsedTime, elapsedMs);
        error = threadLocal_error;
      }
    }
  }

  *elapsedTime = maxElapsedTime;
  HANDLE_ERROR_STMT(error, freeAllocations());

  // Finally, free the allocations.
  freeAllocations(/*sync=*/false);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}

#endif // HAS_CUDA
