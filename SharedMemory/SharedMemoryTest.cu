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
using CacheIndex3D  = uint3;
using VolumeIndex1D = long long;

template< class OutputIndexType, class IndexType >
__device__
inline OutputIndexType to1D(IndexType x, IndexType y, IndexType z, IndexType dimX, IndexType dimY) {
  OutputIndexType i = (OutputIndexType)z * dimY + y;
  return i * dimX + x;
}

__device__ 
void loadSharedData(float* cache, float* data, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ, int K) {
#define NUM_BLOCK_TILES(dim) (dim*2+K-2)/dim

  CacheIndex1D numBlockTilesZ = NUM_BLOCK_TILES(blockDim.z);
  CacheIndex1D numBlockTilesY = NUM_BLOCK_TILES(blockDim.y);
  CacheIndex1D numBlockTilesX = NUM_BLOCK_TILES(blockDim.x);

  for(CacheIndex1D blockTileZ = 0; blockTileZ < numBlockTilesZ; ++blockTileZ) {
    for(CacheIndex1D blockTileY = 0; blockTileY < numBlockTilesY; ++blockTileY) {
      for(CacheIndex1D blockTileX = 0; blockTileX < numBlockTilesX; ++blockTileX) {

        CacheIndex1D       fullBlockZ = blockTileZ * blockDim.z + threadIdx.z;
        CacheIndex1D       fullBlockY = blockTileY * blockDim.y + threadIdx.y;
        CacheIndex1D       fullBlockX = blockTileX * blockDim.x + threadIdx.x;
        CacheIndex1D const cacheIdx   = to1D< CacheIndex1D >(fullBlockX, fullBlockY, fullBlockZ, blockDim.x+K-1, blockDim.y+K-1);
        int const activeInBlock       = (fullBlockZ < blockDim.z + K - 1 && fullBlockY < blockDim.y + K - 1 && fullBlockX < blockDim.x + K - 1);

        fullBlockZ += blockIdx.z * blockDim.z;
        fullBlockY += blockIdx.y * blockDim.y;
        fullBlockX += blockIdx.x * blockDim.x;
        VolumeIndex1D const volumeIndex = to1D< VolumeIndex1D >(fullBlockX, fullBlockY, fullBlockZ, dimX, dimY);
        int const insideGlobal = (fullBlockZ < dimZ && fullBlockY < dimY && fullBlockX < dimX);

        if(activeInBlock && insideGlobal) {
          cache[cacheIdx] = data[volumeIndex];
        }
      }
    }
  }

  // Final synchronization.
  __syncthreads();
}

__device__
void storeResult(float const* cache, float* out, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ, int K) {
  CacheIndex3D const globalIndex {
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z
  };

  VolumeIndex1D volumeIdx = to1D< VolumeIndex1D >(globalIndex.x, globalIndex.y, globalIndex.z, dimX, dimY);

  // Compute the index of the value to write to the global memory, which resides "further back" in shared memory.
  CacheIndex1D featureIndex = (blockDim.x+K-1)*(blockDim.y+K-1)*(blockDim.z+K-1) + to1D< CacheIndex1D >(threadIdx.x,threadIdx.y,threadIdx.z,blockDim.x,blockDim.y);

  if(globalIndex.z < dimZ && globalIndex.y < dimY && globalIndex.x < dimX) {
    out[volumeIdx] = cache[featureIndex];
  }
}


__device__
void statsNDShared(float* features, float* cache, int envSize, int K, int_least64_t* envOffsets) {
  int K2 = K / 2;
  CacheIndex1D const center1D = to1D< CacheIndex1D >(threadIdx.x+K2,threadIdx.y+K2,threadIdx.z+K2,blockDim.x+K-1,blockDim.y+K-1);

  float sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += cache[center1D+envOffsets[voxelIndex]];
  }
  float mean = sum / envSize;

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float stdev = sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = cache[center1D+envOffsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;
  }
  float kurtosis = sum / (envSize * stdev * stdev) - 3.f;

  features[0] = mean;
  features[1] = stdev;
  features[2] = skewness;
  features[3] = kurtosis;

  // Write the output value for each voxel (here: its mean) to a specific position in shared memory first.
  // Do not overwrite the current cache entry since neighboring threads might still access it.
  CacheIndex1D featureIndex = (blockDim.x+K-1)*(blockDim.y+K-1)*(blockDim.z+K-1) + to1D< CacheIndex1D >(threadIdx.x,threadIdx.y,threadIdx.z,blockDim.x,blockDim.y);
  cache[featureIndex] = features[0];
  __syncthreads();
}


__global__
void statisticsKernelNDShared(float* in, float* out, int_least64_t* envOffsets, int K, CacheIndex1D dimX, CacheIndex1D dimY, CacheIndex1D dimZ) {
  extern __shared__ float environmentCache[];
  float features[4];

  for(int z = blockIdx.z * blockDim.z + threadIdx.z; z < dimZ; z += blockDim.z*gridDim.z) {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < dimY; y += blockDim.y*gridDim.y) {
      for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < dimX; x += blockDim.x*gridDim.x) {
        loadSharedData(environmentCache, in, dimX, dimY, dimZ, K);
        statsNDShared(features, environmentCache, K*K*K, K, envOffsets);
        storeResult(environmentCache, out, dimX - K + 1, dimY - K + 1, dimZ - K + 1, K);
      }
    }
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

  if(K > 11) {
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

  if(K > 11) {
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

  // Allocate shared memory for caching the input data (incl. padding) as well 
  // as for storing the decision values (not all features!) which shall be written 
  // to the output later on.
  unsigned int decisionValueMem = threads.x * threads.y * threads.z * sizeof(float);
  unsigned int cacheMemory = (threads.x + K - 1) * (threads.y + K - 1) * (threads.z + K - 1) * sizeof(float);
  unsigned int totalSharedMemory = cacheMemory + decisionValueMem;

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
