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

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Exactly the same as the 1D version.
__device__
void statsND(float* features, float* in, int_least64_t globalIndex, int_least64_t* offsets, int envSize) {
  float sum = 0.f;

  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += in[globalIndex + offsets[voxelIndex]];
  }
  float mean = sum / envSize;

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float stdev = sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;
  }
  float kurtosis = sum / (envSize * stdev * stdev) - 3.f;

  features[0] = mean;
  features[1] = stdev;
  features[2] = skewness;
  features[3] = kurtosis;
}

__global__
void statisticsKernelND(float* in, float* out, int_least64_t* offsets, int K, int_least64_t N, int_least64_t dimX, int_least64_t dimY, int_least64_t dimZ) {
  int K2 = K >> 1;
  int_least64_t globalIndexX = blockIdx.x * blockDim.x + threadIdx.x;
  int_least64_t globalIndexY = blockIdx.y * blockDim.y + threadIdx.y;
  int_least64_t globalIndexZ = blockIdx.z * blockDim.z + threadIdx.z;

  int_least64_t strideX = blockDim.x * gridDim.x;
  int_least64_t strideY = blockDim.y * gridDim.y;
  int_least64_t strideZ = blockDim.z * gridDim.z;

  int_least64_t dimY_withoutPadding = dimY - K + 1;
  int_least64_t dimX_withoutPadding = dimX - K + 1;

  float features[4];

  for(int_least64_t z = globalIndexZ; z < dimZ; z += strideZ) {
    for(int_least64_t y = globalIndexY; y < dimY; y += strideY) {
      for(int_least64_t x = globalIndexX; x < dimX; x += strideX) {
        bool isInPadding = x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2;

        if (!isInPadding) {
          int_least64_t globalIndexWithPadding    = (z * dimY + y) * dimX + x;
          int_least64_t globalIndexWithoutPadding = ((z - K2) * dimY_withoutPadding + (y - K2)) * dimX_withoutPadding + (x - K2);
          statsND(features, in, globalIndexWithPadding, offsets, K*K*K);
          out[globalIndexWithoutPadding] = features[0];
        }
      }
    }
  }
}


// Stuff using shared memory

using CacheIndex1D  = unsigned int;
using VolumeIndex1D = long long;

template< class OutputIndexType, class IndexType >
__device__ inline OutputIndexType to1D(IndexType x, IndexType y, IndexType z, IndexType dimX, IndexType dimY) {
  OutputIndexType i = (OutputIndexType)z * dimY + y;
  return i * dimX + x;
}


template< int Axis > __device__ CacheIndex1D padCacheIdx(CacheIndex1D prim, CacheIndex1D sec, CacheIndex1D ortho, CacheIndex1D K);
template<> __device__ inline CacheIndex1D padCacheIdx</*XY*/0>(CacheIndex1D x, CacheIndex1D y, CacheIndex1D z, CacheIndex1D K) {
  return to1D< CacheIndex1D >(x, y, blockDim.z+z, blockDim.x+K-1, blockDim.y+K-1);
}
template<> __device__ inline CacheIndex1D padCacheIdx</*XZ*/1>(CacheIndex1D x, CacheIndex1D z, CacheIndex1D y, CacheIndex1D K) {
  return to1D< CacheIndex1D >(x, blockDim.y+y, z, blockDim.x+K-1, blockDim.y+K-1);
}
template<> __device__ inline CacheIndex1D padCacheIdx</*YZ*/2>(CacheIndex1D y, CacheIndex1D z, CacheIndex1D x, CacheIndex1D K) {
  return to1D< CacheIndex1D >(blockDim.x+x, y, z, blockDim.x+K-1, blockDim.y+K-1);
}

template< int Axis > __device__ VolumeIndex1D padVolIdx(CacheIndex1D prim, CacheIndex1D sec, CacheIndex1D ortho, CacheIndex1D dimX, CacheIndex1D dimY);
template<> __device__ inline VolumeIndex1D padVolIdx</*XY*/0>(CacheIndex1D x, CacheIndex1D y, CacheIndex1D z, CacheIndex1D dimX, CacheIndex1D dimY) {
  return to1D< VolumeIndex1D >(blockIdx.x*blockDim.x+x, blockIdx.y*blockDim.y+y, blockIdx.z*blockDim.z+blockDim.z+z, dimX, dimY);
}
template<> __device__ inline VolumeIndex1D padVolIdx</*XZ*/1>(CacheIndex1D x, CacheIndex1D z, CacheIndex1D y, CacheIndex1D dimX, CacheIndex1D dimY) {
  return to1D< VolumeIndex1D >(blockIdx.x*blockDim.x+x, blockIdx.y*blockDim.y+blockDim.y+y, blockIdx.z*blockDim.z+z, dimX, dimY);
}
template<> __device__ inline VolumeIndex1D padVolIdx</*YZ*/2>(CacheIndex1D y, CacheIndex1D z, CacheIndex1D x, CacheIndex1D dimX, CacheIndex1D dimY) {
  return to1D< VolumeIndex1D >(blockIdx.x*blockDim.x+blockDim.x+x, blockIdx.y*blockDim.y+y, blockIdx.z*blockDim.z+z, dimX, dimY);
}

template< int Axis, bool CheckIfInRange > 
__device__ inline bool checkIfGlobalIndexWithinVolume(CacheIndex1D prim, CacheIndex1D sec, CacheIndex1D ortho, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  return true;
}
template<> __device__ inline bool checkIfGlobalIndexWithinVolume</*XY*/0, true>(CacheIndex1D x, CacheIndex1D y, CacheIndex1D z, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  return (blockIdx.x * blockDim.x + x < dimX) && (blockIdx.y * blockDim.y + y < dimY) && (blockIdx.z * blockDim.z + z < dimZ);
}
template<> __device__ inline bool checkIfGlobalIndexWithinVolume</*XZ*/1, true>(CacheIndex1D x, CacheIndex1D z, CacheIndex1D y, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  return (blockIdx.x * blockDim.x + x < dimX) && (blockIdx.y * blockDim.y + y < dimY) && (blockIdx.z * blockDim.z + z < dimZ);
}
template<> __device__ inline bool checkIfGlobalIndexWithinVolume</*YZ*/2, true>(CacheIndex1D y, CacheIndex1D z, CacheIndex1D x, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  return (blockIdx.x * blockDim.x + x < dimX) && (blockIdx.y * blockDim.y + y < dimY) && (blockIdx.z * blockDim.z + z < dimZ);
}


template< int Axis, bool CheckIfInRange = false >
__device__
inline void loadPaddingArea(
  float* cache, float const* data, 
  CacheIndex1D thread1D, CacheIndex1D K, 
  CacheIndex1D padDimX, 
  CacheIndex1D volDimX, CacheIndex1D volDimY, CacheIndex1D volDimZ
)
{
  // Here, we consider a 2D "area" of threads which load padding voxels.
  // A loop repeats this for the number of areas, equals to the padding required.
  CacheIndex1D const vertLocal = thread1D / padDimX;
  CacheIndex1D const horzLocal = thread1D % padDimX;
  for(CacheIndex1D ortho = 0; ortho < K-1; ++ortho) {
    CacheIndex1D  const cacheIndex  = padCacheIdx< Axis >(horzLocal, vertLocal, ortho, K);
    VolumeIndex1D const volumeIndex = padVolIdx<   Axis >(horzLocal, vertLocal, ortho, volDimX, volDimY);

    if(checkIfGlobalIndexWithinVolume<Axis, CheckIfInRange>(horzLocal,vertLocal,ortho,volDimX,volDimY,volDimZ)) {
      cache[cacheIndex] = data[volumeIndex];
    }
  }
}

template< bool CheckIfInRange >
__device__
inline void loadPadding(float* cache, float const* data, int K, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  // These prefixes group threads together:
  // Before this function is called, each thread (we assume 1024 threads) loads a single voxel to shared memory.
  // Here, we load the remaining padding regions with the following groupings, where we assume a thread block has
  // a shape of (x=32, y=8, z=4) threads and the algorithm requires a padding of p voxels in each dimension.
  //   o The first group of 32 x 8 = 256 threads loads the XY padding (in the "back" z region). Repeated p times.
  //   o The second group of 32 x (4+p) threads loads the XZ padding (in the "back" y region). Repeated p times.
  //   o The third group of (8+p) x (4+p) threads loads the YZ padding (in the "back" x region). Repeated p times.
  // As we allow a maximum value of K=11 (padding of 10 voxels) due to hardware resources, the above grouping remains
  // within 1024 threads.
  unsigned int const paddingGroupPrefix[3] = {
     blockDim.x*blockDim.y, 
     blockDim.x*blockDim.y + blockDim.x*(blockDim.z+K-1), 
     blockDim.x*blockDim.y + blockDim.x*(blockDim.z+K-1) + (blockDim.y+K-1)*(blockDim.z+K-1)
  };

  // Depending on the (1D) thread index, execute the different padding load loops.
  unsigned int const thread1D = to1D<unsigned int>(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y);

  if (thread1D < paddingGroupPrefix[0]) {
    loadPaddingArea<0, CheckIfInRange>(cache, data, thread1D,                         K, blockDim.x,     dimX, dimY, dimZ);
  } else if(thread1D < paddingGroupPrefix[1]) {
    loadPaddingArea<1, CheckIfInRange>(cache, data, thread1D - paddingGroupPrefix[0], K, blockDim.x,     dimX, dimY, dimZ);
  } else if(thread1D < paddingGroupPrefix[2]) {
    loadPaddingArea<2, CheckIfInRange>(cache, data, thread1D - paddingGroupPrefix[1], K, blockDim.y+K-1, dimX, dimY, dimZ);
  }
}


__device__ 
void loadSharedData(float* cache, float const* data, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ, int K) {
  CacheIndex1D  const cacheIndex  = to1D<  CacheIndex1D >(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x + K - 1, blockDim.y + K - 1);
  VolumeIndex1D const volumeIndex = to1D< VolumeIndex1D >(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z, dimX, dimY);

  if(blockIdx.x + 1 == gridDim.x || blockIdx.y + 1 == gridDim.y || blockIdx.z + 1 == gridDim.z) {
    // Special case for the last block per dimension: This block is typically not fully filled, 
    // thus we need additional bounds checking for the thread accesses to the global volume.
    if(checkIfGlobalIndexWithinVolume<0,true>(threadIdx.x,threadIdx.y,threadIdx.z,dimX,dimY,dimZ)) {
      cache[cacheIndex] = data[volumeIndex];
    }
    loadPadding< true >(cache, data, K, dimX, dimY, dimZ);
  } else {
    // In case we are not in the very last block, the blocks are always fully filled.
    // Thus, we do not need additional bounds checking here.
    cache[cacheIndex] = data[volumeIndex];
    loadPadding< false >(cache, data, K, dimX, dimY, dimZ);
  }

  // Final synchronization.
  __syncthreads();
}


__device__
void statsNDShared(float* features, float const* cache, int envSize, int K, int_least64_t const* envOffsets) {
  int const K2 = K >> 1;
  CacheIndex1D const center1D = to1D< CacheIndex1D >(threadIdx.x+K2,threadIdx.y+K2,threadIdx.z+K2,blockDim.x+K-1,blockDim.y+K-1);

  float sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += cache[center1D+envOffsets[voxelIndex]];
  }
  float const mean = sum / envSize;

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float const stdev = sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float const skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;
  }
  float const kurtosis = sum / (envSize * stdev * stdev) - 3.f;

  features[0] = mean;
  features[1] = stdev;
  features[2] = skewness;
  features[3] = kurtosis;
}


__global__
void statisticsKernelNDShared(float const* in, float* out, int_least64_t const* envOffsets, int K, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  extern __shared__ float cache[];
  float features[4];

  loadSharedData(cache, in, dimX, dimY, dimZ, K);
  statsNDShared(features, cache, K*K*K, K, envOffsets);

  CacheIndex1D const globalZ = blockIdx.z * blockDim.z + threadIdx.z;
  CacheIndex1D const globalY = blockIdx.y * blockDim.y + threadIdx.y;
  CacheIndex1D const globalX = blockIdx.x * blockDim.x + threadIdx.x;
  CacheIndex1D const padding = K - 1;

  if(globalZ < dimZ - padding && globalY < dimY - padding && globalX < dimX - padding) {
    out[to1D< VolumeIndex1D >(globalX, globalY, globalZ, dimX - padding, dimY - padding)] = features[0];
  }
}

// ======================================================================================================================

cudaError_t launchKernelNDWithoutShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  unsigned int dimX, unsigned int dimY, unsigned int dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  if(K > 9) {
    // While hardware resources (at least the ones which we consider, cm_52 or higher)
    // would permit using K==11, we limt ourselves to at max K==9 to achieve maximum occupancy.
    return cudaError_t::cudaErrorLaunchOutOfResources;
  }

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  HANDLE_ERROR(error);

  int pad = K - 1;
  int_least64_t N            = dimX * dimY * dimZ;
  int_least64_t sizeInBytes  = N * sizeof(float);
  int_least64_t sizeOutBytes = (dimX - pad) * (dimY - pad) * (dimZ - pad) * sizeof(float);
  int_least64_t offsetBytes  = K * K * K * sizeof(int_least64_t);

  float* device_in;
  error = cudaMalloc((void**)&device_in, sizeInBytes);
  HANDLE_ERROR(error);
  
  float* device_out;
  error = cudaMalloc((void**)&device_out, sizeInBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in));
  
  int_least64_t* device_offsets;
  error = cudaMalloc((void**)&device_offsets, offsetBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in); cudaFree(device_out));

  error = cudaMemcpy(device_in,           in, sizeInBytes, cudaMemcpyHostToDevice);
  error = cudaMemcpy(device_offsets, offsets, offsetBytes, cudaMemcpyHostToDevice);
  HANDLE_ERROR(error);

  cudaEvent_t start, stop;
  float elapsedMs = -1;

  unsigned int blocksX = (dimX + threads.x - 1) / threads.x;
  unsigned int blocksY = (dimY + threads.y - 1) / threads.y;
  unsigned int blocksZ = (dimZ + threads.z - 1) / threads.z;
  dim3 blocks(blocksX, blocksY, blocksZ);

  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);

  statisticsKernelND<<< blocks, threads >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  error = cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  error = cudaEventElapsedTime(&elapsedMs, start, stop);
  elapsedMillis.push_back(elapsedMs);
  
  error = cudaMemcpy(out, device_out, sizeOutBytes, cudaMemcpyDeviceToHost);

  error = cudaFree(device_in);
  error = cudaFree(device_out);
  error = cudaFree(device_offsets);
  HANDLE_ERROR(error);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}


cudaError_t launchKernelNDWithShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  unsigned int dimX, unsigned int dimY, unsigned int dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  if(K > 9) {
    // While hardware resources (at least the ones which we consider, cm_52 or higher)
    // would permit using K==11, we limt ourselves to at max K==9 to achieve maximum occupancy.
    return cudaError_t::cudaErrorLaunchOutOfResources;
  }

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  HANDLE_ERROR(error);

  int pad = K - 1;
  int_least64_t N            = dimX * dimY * dimZ;
  int_least64_t sizeInBytes  = N * sizeof(float);
  int_least64_t sizeOutBytes = (dimX - pad) * (dimY - pad) * (dimZ - pad) * sizeof(float);
  int_least64_t offsetBytes  = K * K * K * sizeof(int_least64_t);

  float* device_in;
  error = cudaMalloc((void**)&device_in, sizeInBytes);
  HANDLE_ERROR(error);
  
  float* device_out;
  error = cudaMalloc((void**)&device_out, sizeInBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in));
  
  int_least64_t* device_offsets;
  error = cudaMalloc((void**)&device_offsets, offsetBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in); cudaFree(device_out));

  error = cudaMemcpy(device_in,           in, sizeInBytes, cudaMemcpyHostToDevice);
  error = cudaMemcpy(device_offsets, offsets, offsetBytes, cudaMemcpyHostToDevice);
  HANDLE_ERROR(error);

  cudaEvent_t start, stop;
  float elapsedMs = -1;

  // --------------------------------------------------------------------------------------------------
  // Launch 3D grid with shared memory access kernel.

  unsigned int blocksX = (dimX + threads.x - 1) / threads.x;
  unsigned int blocksY = (dimY + threads.y - 1) / threads.y;
  unsigned int blocksZ = (dimZ + threads.z - 1) / threads.z;
  dim3 blocks(blocksX, blocksY, blocksZ);

  // Allocate shared memory for caching the input data (incl. padding).
  unsigned int totalSharedMemory = (threads.x + K - 1) * (threads.y + K - 1) * (threads.z + K - 1) * sizeof(float);

  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);

  statisticsKernelNDShared<<< blocks, threads, totalSharedMemory >>>(device_in, device_out, device_offsets, K, dimX, dimY, dimZ);

  error = cudaPeekAtLastError();
  HANDLE_ERROR(error);
  error = cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  error = cudaEventElapsedTime(&elapsedMs, start, stop);
  elapsedMillis.push_back(elapsedMs);
  HANDLE_ERROR(error);

  // --------------------------------------------------------------------------------------------------
  error = cudaMemcpy(out, device_out, sizeOutBytes, cudaMemcpyDeviceToHost);

  error = cudaFree(device_in);
  error = cudaFree(device_out);
  error = cudaFree(device_offsets);
  HANDLE_ERROR(error);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}
