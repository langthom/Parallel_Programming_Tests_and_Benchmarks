
#include <algorithm>
#include <array>
#include <cassert>
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

int main(int argc, char** argv) {
  int K = 3;
  int K2 = K >> 1;
  int E = 3;

  size_t dimZ = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimX = 1ull << (E + 1);
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

  std::vector< float > groundTruth(N, 0);
  for (size_t i = 0; i < N; ++i) {
    size_t posZ = i / (dimX * dimY);
    size_t it   = i - posZ * dimY * dimX;
    size_t posY = it / dimX;
    size_t posX = it % dimX;

    if (posX < K2 || posY < K2 || posZ < K2 || posX > dimX - 1 - K2 || posY > dimY - 1 - K2 || posZ > dimZ - 1 - K2) {
      groundTruth[i] = 0.f;
      continue;
    }

    float sum = 0.f;
    for (size_t j = 0; j < K; ++j) {
      sum += data[i + offsets[j]];
    }

    groundTruth[i] = sum / (K * K * K);
  }

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

  int cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ);
  if (cuda_kernel_error != 0) {
    std::cout << "SOME ERROR HAPPENED\n";
    return -1;
  }

  float maxCUDAError = 0.f;
  auto inIt = groundTruth.begin(), outIt = host_output.begin();
  for (; inIt != groundTruth.end(); ++inIt, ++outIt) {
    maxCUDAError = std::fmaxf(maxCUDAError, std::fabsf(*inIt - *outIt));
  }
  std::cout << "Max error with CUDA execution is: " << maxCUDAError << '\n';
#endif

  // Call the OpenCL kernel


  return 0;
}