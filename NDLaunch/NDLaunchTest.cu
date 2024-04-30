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


// 3D kernel for computing statistic moments.
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

cudaError_t launchKernelND(
  float* out, float* in, int_least64_t N, 
  int_least64_t* offsets, int K, 
  int_least64_t dimX, int_least64_t dimY, int_least64_t dimZ, 
  dim3 threads,
  float* elapsedTime, 
  int deviceID)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  // The following block is exactly the same as our 1D launch in the CommonKernels header/implementation.

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  HANDLE_ERROR(error);

  int pad = K - 1;
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
  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);

  // Change: Multi-dimensional blocks covering the entire grid.
  unsigned int blocksX = (dimX + threads.x - 1) / threads.x;
  unsigned int blocksY = (dimY + threads.y - 1) / threads.y;
  unsigned int blocksZ = (dimZ + threads.z - 1) / threads.z;
  dim3 blocks(blocksX, blocksY, blocksZ);

  statisticsKernelND<<< blocks, threads >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  // Same as 1D again.
  error = cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  error = cudaEventElapsedTime(elapsedTime, start, stop);
  
  error = cudaMemcpy(out, device_out, sizeOutBytes, cudaMemcpyDeviceToHost);

  error = cudaFree(device_in);
  error = cudaFree(device_out);
  error = cudaFree(device_offsets);
  HANDLE_ERROR(error);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}
