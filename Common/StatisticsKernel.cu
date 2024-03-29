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

#include "StatisticsKernel.h"

#ifdef HAS_CUDA

__device__
void stats(float* features, float* in, int64_t globalIndex, int64_t* offsets, int envSize) {
  // This is not an efficient or even numerically stable way to compute the centralized 
  // statistical moments, but it does a few more computations and thus put some mild
  // computational load on the device.

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
void statisticsKernel(float* in, float* out, int64_t* offsets, int K, int64_t N, int64_t dimX, int64_t dimY, int64_t dimZ) {
  int K2 = K >> 1;
  int64_t globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride      = blockDim.x * gridDim.x;

  int64_t dimY_withoutPadding = dimY - K + 1;
  int64_t dimX_withoutPadding = dimX - K + 1;

  float features[4];

  for(int64_t i = globalIndex; i < N; i += stride) {
    int64_t z = i / (dimY * dimX);
    int64_t j = i - z * dimY * dimX;
    int64_t y = j / dimX;
    int64_t x = j % dimX;

    bool isInPadding = x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2;

    if (!isInPadding) {
      stats(features, in, i, offsets, K*K*K);
      int64_t globalIndexWithoutPadding  = ((z - K2) * dimY_withoutPadding + (y - K2)) * dimX_withoutPadding + (x - K2);
      out[globalIndexWithoutPadding] = features[0];
    }
  }
}

#endif // HAS_CUDA
