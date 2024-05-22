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
          //out[globalIndexWithoutPadding] = features[0];
          out[globalIndexWithoutPadding] = in[globalIndexWithPadding];
        }
      }
    }
  }
}


// Stuff using shared memory

using CacheIndex1D  = unsigned int;
using CacheIndex3D  = uint3;
using VolumeIndex1D = long long;
using VolumeIndex3D = longlong3;


template< class IndexVectorType, class IndexType = decltype(IndexVectorType{}.x) > 
__device__ 
IndexType to1D(IndexVectorType coord3D, IndexVectorType const dimsOffsets) {
  coord3D.x *= dimsOffsets.x;
  coord3D.y *= dimsOffsets.y;
  coord3D.z *= dimsOffsets.z;
  return coord3D.x + coord3D.y + coord3D.z;
}

__device__ 
void loadSharedData(float* cache, float* data, int dimX, int dimY, int dimZ, int K) {
#define NUM_BLOCK_TILES(dim) (dim*2+K-2)/dim

  VolumeIndex3D const volumeOffsets{ 1, dimX, dimX * dimY };
  CacheIndex3D const fullBlockShape{ blockDim.x+K-1, blockDim.y+K-1, blockDim.z+K-1 };
  CacheIndex3D const cacheOffsets{ 1, fullBlockShape.x, fullBlockShape.x * fullBlockShape.y };

  int3 const numBlockTiles{ NUM_BLOCK_TILES(blockDim.x), NUM_BLOCK_TILES(blockDim.y), NUM_BLOCK_TILES(blockDim.z) };

  for(CacheIndex1D blockTileZ = 0; blockTileZ < numBlockTiles.z; ++blockTileZ) {
    for(CacheIndex1D blockTileY = 0; blockTileY < numBlockTiles.y; ++blockTileY) {
      for(CacheIndex1D blockTileX = 0; blockTileX < numBlockTiles.x; ++blockTileX) {
        CacheIndex1D zWithinFullBlock = blockTileZ * blockDim.z + threadIdx.z;
        CacheIndex1D yWithinFullBlock = blockTileY * blockDim.y + threadIdx.y;
        CacheIndex1D xWithinFullBlock = blockTileX * blockDim.x + threadIdx.x;

        VolumeIndex1D globalZ = (VolumeIndex1D)(blockIdx.z * blockDim.z + zWithinFullBlock);
        VolumeIndex1D globalY = (VolumeIndex1D)(blockIdx.y * blockDim.y + yWithinFullBlock);
        VolumeIndex1D globalX = (VolumeIndex1D)(blockIdx.x * blockDim.x + xWithinFullBlock);

        int activeInBlock = (zWithinFullBlock < fullBlockShape.z && yWithinFullBlock < fullBlockShape.y && xWithinFullBlock < fullBlockShape.x);
        int insideGlobal  = (globalZ < dimZ && globalY < dimY && globalX < dimX);

        if(activeInBlock && insideGlobal) {
          cache[to1D({xWithinFullBlock,yWithinFullBlock,zWithinFullBlock}, cacheOffsets)] = data[to1D({globalX,globalY,globalZ}, volumeOffsets)];
        }
      }
    }
  }

  // Final synchronization.
  __syncthreads();
}

__device__
void storeResult(float const* cache, float* out, int dimX, int dimY, int dimZ, int K) {
  int K2 = K / 2;
  CacheIndex3D const fullBlockShape{ blockDim.x+K-1, blockDim.y+K-1, blockDim.z+K-1 };
  CacheIndex3D const cacheOffsets{ 1, fullBlockShape.x, fullBlockShape.x * fullBlockShape.y };
  VolumeIndex3D const volumeOffsets{ 1, dimX, dimX * dimY };

  VolumeIndex3D const globalIndex {
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z
  };

  CacheIndex3D const localIndex{
    (CacheIndex1D)threadIdx.x + K2,
    (CacheIndex1D)threadIdx.y + K2,
    (CacheIndex1D)threadIdx.z + K2,
  };

  if(globalIndex.z < dimZ && globalIndex.y < dimY && globalIndex.x < dimX) {
    out[to1D(globalIndex, volumeOffsets)] = cache[to1D(localIndex, cacheOffsets)];
  }
}


__device__
void statsNDShared(float* features, float* cache, int envSize, int K, int_least64_t* envOffsets) {
  int K2 = K / 2;
  CacheIndex3D const cacheOffsets{ 1, blockDim.x+K-1, (blockDim.x+K-1) * (blockDim.y+K-1) };
  CacheIndex3D const center3D{
    (CacheIndex1D)threadIdx.x + K2,
    (CacheIndex1D)threadIdx.y + K2,
    (CacheIndex1D)threadIdx.z + K2,
  };

  CacheIndex1D const center1D = to1D(center3D, cacheOffsets);

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
}


__global__
void statisticsKernelNDShared(float* in, float* out, int_least64_t* envOffsets, int K, int dimX, int dimY, int dimZ) {
  extern __shared__ float environmentCache[];
  float features[4];

#if 1
  {{{
      int_least64_t z = blockIdx.z * blockDim.z + threadIdx.z;
      int_least64_t y = blockIdx.y * blockDim.y + threadIdx.y;
      int_least64_t x = blockIdx.x * blockDim.x + threadIdx.x;
#else
  for(int z = blockIdx.z * blockDim.z + threadIdx.z; z < dimZ; z += blockDim.z*gridDim.z) {
    for(int y = blockIdx.y * blockDim.y + threadIdx.y; y < dimY; y += blockDim.y*gridDim.y) {
      for(int x = blockIdx.x * blockDim.x + threadIdx.x; x < dimX; x += blockDim.x*gridDim.x) {
#endif

        loadSharedData(environmentCache, in, dimX, dimY, dimZ, K);

        //statsNDShared(features, environmentCache, K*K*K, K, envOffsets);

        storeResult(environmentCache, out, dimX - K + 1, dimY - K + 1, dimZ - K + 1, K);
      }
    }
  }
}

// ======================================================================================================================

cudaError_t launchKernelNDWithoutShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  int_least64_t dimX, int_least64_t dimY, int_least64_t dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  if(K > 9) {
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


#include <iostream>
cudaError_t launchKernelNDWithShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  int_least64_t dimX, int_least64_t dimY, int_least64_t dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  if(K > 9) {
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

  int_least64_t sharedMem = (threads.x + K - 1) * (threads.y + K - 1) * (threads.z + K - 1) * sizeof(float);

  std::cout << "\n(K = " << K << ") shared mem: " << sharedMem << " bytes\n";

  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);

  statisticsKernelNDShared<<< blocks, threads, sharedMem >>>(device_in, device_out, device_offsets, K, dimX, dimY, dimZ);

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
