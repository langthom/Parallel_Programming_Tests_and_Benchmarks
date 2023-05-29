
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef HAS_CUDA
#include "MaskBasedProcessing_OpenCL_vs_CUDA.h"
#endif

float formatMemory(float N, std::string& unit) {
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


inline std::array< float, 4 > stats(float* buffer, size_t* offsets, int index, int K) {
  int envSize = K * K * K;
  std::vector< float > env(envSize);
  for (int j = 0; j < envSize; ++j) {
    env[j] = buffer[index + offsets[j]];
  }

  std::array< float, 4 > features;

  float count = 0, mean = 0, M2 = 0, M3 = 0, M4 = 0;
  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxel = env[voxelIndex];
    float n1 = count;
    count += 1;
    float delta = voxel - mean;
    float delta_n = delta / count;
    float delta_n2 = delta_n * delta_n;
    float term1 = delta * delta_n * n1;
    mean += delta_n;
    M4 += term1 * delta_n2 * (count * count - 3 * count + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
    M3 += term1 * delta_n * (count - 2) - 3 * delta_n * M2;
    M2 += term1;
  }

  features[0] = mean;
  features[1] = std::powf(M2 / count, 0.5f);
  features[2] = M2 < 1e-7 ? 0.0 : (M3 * pow(count, 0.5f) / pow(M2, 1.5f));
  features[3] = M2 < 1e-7 ? -3.0 : ((M4 * count) / (M2 * M2) - 3);
  return features;
}



int main(int argc, char** argv) {
  int K = 3;
  int K2 = K >> 1;
  int E = 9;

  float memInGB = 0.25; // need this two times (in/out) and tmp memory!

  size_t dimX = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimZ = (size_t)(memInGB * (1ull << 30)) / (sizeof(float)*dimY*dimX);
  size_t N = dimZ * dimY * dimX;
  size_t sizeInBytes = N * sizeof(float);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using " << N << " elements, corresponding to " << _N << " " << unit << '\n';

  // Prepare the random input data.
  std::random_device rd;
  std::mt19937 gen{ rd() };
  std::uniform_real_distribution< float > dist{ 0.f, 100.f };

  std::vector< float > data(N, 0);
  std::vector< float > host_output(N, 0);
  std::for_each(data.begin(), data.end(), [&](float& f) {f = dist(gen); });

  // Prepare the ground truth results on CPU.
  std::vector< size_t > offsets(K*K*K, 0);
  size_t _o = 0;
  for (int _z = -K2; _z <= K2; ++_z) {
    for (int _y = -K2; _y <= K2; ++_y) {
      for (int _x = -K2; _x <= K2; ++_x) {
        offsets[_o++] = (_z * dimY + _y) * dimX + _x;
      }
    }
  }

  float* dataPtr = data.data();
  std::vector< float > groundTruth(N, 0);

  auto cpuStart = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
  for (long long i = 0; i < N; ++i) {
    size_t posZ = i / (dimX * dimY);
    size_t it   = i - posZ * dimY * dimX;
    size_t posY = it / dimX;
    size_t posX = it % dimX;

    if (posX < K2 || posY < K2 || posZ < K2 || posX > dimX - 1 - K2 || posY > dimY - 1 - K2 || posZ > dimZ - 1 - K2) {
      groundTruth[i] = 0.f;
    } else {
      auto features = stats(dataPtr, offsets.data(), i, K);
      groundTruth[i] = features[0];
    }
  }

  auto cpuEnd = std::chrono::high_resolution_clock::now();
  auto cpuDuration = std::chrono::duration_cast< std::chrono::milliseconds >(cpuEnd - cpuStart).count();
  std::cout << "CPU execution took " << cpuDuration << " milliseconds.\n";

  // Call the CUDA kernel
#ifdef HAS_CUDA

  int nr_gpus = 0;
  size_t* availableMemoryPerDevice = NULL;
  size_t* totalMemoryPerDevice = NULL;
  getGPUInformation(&nr_gpus, &availableMemoryPerDevice, &totalMemoryPerDevice);

  std::cout << "[INFO] Detected " << nr_gpus << " GPUs that can run CUDA.\n";
  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    std::string freeUnit, totalUnit;
    float freeMem  = formatMemory(availableMemoryPerDevice[gpuID], freeUnit);
    float totalMem = formatMemory(    totalMemoryPerDevice[gpuID], totalUnit);
    float frac     = (double)availableMemoryPerDevice[gpuID] / (double)totalMemoryPerDevice[gpuID];
    std::cout << "[INFO]   Device " << gpuID << " has " << std::setprecision(3) << freeMem << ' ' << freeUnit << " free memory out of " << std::setprecision(3) << totalMem << ' ' << totalUnit << " (-> " << std::setprecision(4) << frac*100.f << " %)\n";
  }

  auto call_cuda_kernel = [&](int threadsPerBlock) {
    float elapsedTimeInMilliseconds = -1;
    int cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, threadsPerBlock);
    if (cuda_kernel_error != 0) {
      std::cout << "SOME ERROR HAPPENED\n";
    }

    std::cout << " - Execution of kernel (" << std::setw(3) << threadsPerBlock << " threads/block) took " << std::setprecision(5) << elapsedTimeInMilliseconds << " milliseconds.\n";

    float maxCUDAError = 0.f;
    auto inIt = groundTruth.begin(), outIt = host_output.begin();
    for (; inIt != groundTruth.end(); ++inIt, ++outIt) {
      maxCUDAError = std::fmaxf(maxCUDAError, std::fabsf(*inIt - *outIt));
      if (maxCUDAError > 0) {
        std::cerr << *inIt << " - " << *outIt << '\n'; break;
      }
    }
    if (maxCUDAError > 1e-5) {
      std::cout << "Max error with CUDA execution is: " << maxCUDAError << '\n';
    }
  };

  for (int threadsPerBlock : {1, 4, 16, 64, 256, 512}) {
    call_cuda_kernel(threadsPerBlock);
  }

#endif

  // Call the OpenCL kernel


  return 0;
}