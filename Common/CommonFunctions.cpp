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

#include "CommonFunctions.h"

#include <array>
#include <chrono>
#include <cmath> // std::sqrtf

std::array< float, 4 > stats(float* buffer, int64_t* offsets, int64_t index, int K) {
  int envSize = K * K * K;
  std::vector< float > env(envSize);
  for (int j = 0; j < envSize; ++j) {
    env[j] = buffer[index + offsets[j]];
  }

  std::array< float, 4 > features;

  float sum = 0.f;

  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += env[voxelIndex];
  }
  float mean = sum / envSize;

  sum = 0.f;
  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = env[voxelIndex] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float stdev = std::sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = env[voxelIndex] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = env[voxelIndex] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;
  }
  float kurtosis = sum / (envSize * stdev * stdev) - 3.f;

  features[0] = mean;
  features[1] = stdev;
  features[2] = skewness;
  features[3] = kurtosis;
  return features;
}

double cpuKernel(float* groundTruth, float* dataPtr, int64_t* offsets, int64_t N, int K, int64_t dimX, int64_t dimY, int64_t dimZ, bool parallel) {
  int K2 = K / 2;
  auto cpuStart = std::chrono::high_resolution_clock::now();

  int64_t dimY_withoutPadding = dimY - K + 1;
  int64_t dimX_withoutPadding = dimX - K + 1;

  #pragma omp parallel for if(parallel)
  for (int64_t i = 0; i < N; ++i) {
    int64_t posZ = i / (dimX * dimY);
    int64_t it   = i - posZ * dimY * dimX;
    int64_t posY = it / dimX;
    int64_t posX = it % dimX;
    bool insidePadding = posX < K2 || posY < K2 || posZ < K2 || posX > dimX - 1 - K2 || posY > dimY - 1 - K2 || posZ > dimZ - 1 - K2;

    if (!insidePadding) {
      auto features = stats(dataPtr, offsets, i, K);
      int64_t indexWithoutPadding = ((posZ - K2) * dimY_withoutPadding + (posY - K2)) * dimX_withoutPadding + (posX - K2);
      groundTruth[indexWithoutPadding] = features[0];
    }
  }

  auto cpuEnd = std::chrono::high_resolution_clock::now();
  double cpuDuration = std::chrono::duration< double, std::milli >(cpuEnd - cpuStart).count(); // fractional milliseconds;
  return cpuDuration;
}

float computeMaxError(float* groundTruth, float* computedOutput, int64_t bufferSize) {
  float* inIt  = groundTruth;
  float* outIt = computedOutput;
  float maxError = 0.f;

  #pragma omp parallel
  {
    float maxErrorForThisThread = 0.f;

    #pragma omp for nowait
    for (int64_t i = 0; i < bufferSize; ++i) {
      float absError = std::fabsf(inIt[i] - outIt[i]);
      maxErrorForThisThread = std::fmaxf(maxErrorForThisThread, absError);
    }

    #pragma omp critical
    maxError = std::fmaxf(maxError, maxErrorForThisThread);
  }

  return maxError;
}

double formatMemory(double N, std::string& unit) {
  std::array< const char*, 4 > fmts{
    "B", "kiB", "MiB", "GiB"
  };
  int fmtI = 0;
  for (; N >= 1024 && fmtI < fmts.size(); ++fmtI) {
    N /= 1024.f;
  }

  unit = fmts[fmtI];
  return N;
}

std::vector< int64_t > computeMaskOffsets(int envSize, std::size_t dimY, std::size_t dimX) {
  int64_t voxelsInEnvironment = static_cast< int64_t >(envSize) * envSize * envSize;
  std::vector< int64_t > offsets(voxelsInEnvironment, 0);
  int K2 = envSize / 2;
  int64_t _o = 0;

  for (int64_t _z = -K2; _z <= K2; ++_z) {
    for (int64_t _y = -K2; _y <= K2; ++_y) {
      for (int64_t _x = -K2; _x <= K2; ++_x) {
        offsets[_o++] = (_z * dimY + _y) * dimX + _x;
      }
    }
  }

  return offsets;
}

