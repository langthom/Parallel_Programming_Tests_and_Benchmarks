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
#define NOMINMAX

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

#include "../Common/CommonFunctions.h"
#include "../Common/CommonKernels.h"


constexpr int minLogThreadsPerBlock() {
  // Yield the minimum (logarithmized) number of threads per block.
  // Since we experienced time-outs if we only use 1 thread per block,
  // we use a minimum of 2**4 == 16 threads per block.
  return 4;
}

void readData(float* buf, size_t N) {
  // Simulate some data reading process. This is meant to replace
  // any hard-coded values due to the memory mapping business below.
  std::fill_n(buf, N, 1.f);
}

void benchmark(std::ostream& out, double memInGiB, int K, bool printSpecs, int inMaxThreadsPerBlock) {
  int K2 = K >> 1;
  int E = 9;

  size_t dimX = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimZ = (size_t)(memInGiB * (1ull << 30)) / (2/* in and output */ * sizeof(float) * dimY * dimX);
  size_t N = dimZ * dimY * dimX;
  size_t sizeInBytes = N * sizeof(float);
  size_t envSize = static_cast<size_t>(K) * K * K;

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  size_t offsetBytes = offsets.size() * sizeof(int);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using " << N << " elements, corresponding to " << (2 * _N) << " " << unit << " overall; Using regions of size " << K << " x " << K << " x " << K << ".\n";

  size_t pad = K2 * 2;
  size_t sizeWithoutPadding = (dimX - pad) * (dimY - pad) * (dimZ - pad) * sizeof(float);
  std::vector< float > groundTruth(sizeWithoutPadding, 0);

  // Baseline: Single threaded and OpenMP-parallelized CPU implementation
  {
    std::vector< float > data(N, 0);
    readData(data.data(), N);
    double cpuDuration = cpuKernel(groundTruth.data(), data.data(), offsets.data(), N, K, dimX, dimY, dimZ, false);
    out << memInGiB << ',' << K << ',' << cpuDuration;
    cpuDuration = cpuKernel(groundTruth.data(), data.data(), offsets.data(), N, K, dimX, dimY, dimZ, true);
    out << memInGiB << ',' << K << ',' << cpuDuration;
  }

  // Call the CUDA kernel
#ifdef HAS_CUDA
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  if (error != cudaError_t::cudaSuccess) {
    std::cout << "[CUDA] Error during GPU information retrieval, error was: " << cudaGetErrorString(error);
    return;
  } else {
    std::vector< float > data(N, 0);
    std::vector< float > host_output(groundTruth.size(), 0);
    readData(data.data(), N);

    auto call_cuda_kernel = [&](int deviceID, int threadsPerBlock) -> bool {
      float elapsedTimeInMilliseconds = -1;
      cudaError_t cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, deviceID, threadsPerBlock);
      if (cuda_kernel_error != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]  Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'\n";
        return false;
      }

      out << ',' << threadsPerBlock << ',' << elapsedTimeInMilliseconds;

      float maxCUDAError = computeMaxError(groundTruth.data(), host_output.data(), K, dimX, dimY, dimZ);
      if (maxCUDAError > 1e-5) {
        std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
        return false;
      }
      return true;
    };

    int minThreadsPerBlock = std::min(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    int maxThreadsPerBlock = std::max(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      for (int threadsPerBlock = minThreadsPerBlock; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
        if (!call_cuda_kernel(gpuID, 1ull << threadsPerBlock)) {
          goto AFTER_CUDA;
        }
      }

      int maxPotentialCUDABlockSize = 1;
      cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, gpuID);
      if (blockSizeRetErr != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
      } else {
        call_cuda_kernel(gpuID, maxPotentialCUDABlockSize);
      }
    }
  }

AFTER_CUDA:;

#endif
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " <path/to/output/benchmark/csv>  <max-mem-in-GiB>  <num-percentages>  <Kmin>  <Kmax>  <threadsPerBlock>\n\n";
    std::cerr << "  <path/to/output/benchmark/csv>  File path to the benchmark file that will be generated.\n";
    std::cerr << "  <max-mem-in-GiB>                Maximum memory to be used on GPU in GiB (float). If the device does not have sufficient memory, this will be reduced automatically.\n";
    std::cerr << "  <num-percentages>               Number of percentages of the max memory to sample. Must be bigger in [1..100], e.g. setting 2 will result in using 100% and 50%.\n";
    std::cerr << "  <Kmin>                          Minimum environment size, must be bigger than one and odd.\n";
    std::cerr << "  <Kmax>                          Maximum environment size, must be bigger than one and odd.\n";
    std::cerr << "  <threadsPerBlock>               Max (logarithmized) number of threads per block, e.g., 9 == 512 threads per block. The threads per block will get incremented by two each loop, i.e., [1, 4, ...]\n";
    return EXIT_FAILURE;
  }

  auto parseInt = [](char const* arg) -> int {
    int value;
    std::istringstream(arg) >> value;
    return value;
  };

  auto parseDouble = [](char const* arg) -> int {
    double value;
    std::istringstream(arg) >> value;
    return value;
  };

  std::string benchmarkLogFile = argv[1];
  double maxMemoryGiB = parseDouble(argv[2]);
  int numPercentages = parseInt(argv[3]);
  int _Kmin = parseInt(argv[4]);
  int _Kmax = parseInt(argv[5]);
  int Kmin = std::min(_Kmin, _Kmax);
  int Kmax = std::max(_Kmin, _Kmax);
  int maxThreadsBlock = parseInt(argv[6]);


  double maxMemoryInGB = getMaxMemoryInGiB(maxMemoryGiB, 0.9, false);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n";

  std::ofstream log(benchmarkLogFile);

  bool firstRun = true;
  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    for (int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, percentage / 100.0 * maxMemoryInGB, K, firstRun, maxThreadsBlock);
      log << std::string(100, '=') << '\n';
      firstRun = false;

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return EXIT_SUCCESS;
}