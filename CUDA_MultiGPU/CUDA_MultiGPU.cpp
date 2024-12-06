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

#include "../Common/MultiGPUExecution.h"
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

void benchmark(std::ostream& out, double memInGiB, int K, int inMaxThreadsPerBlock, bool measureSingleThreadedCPU) {
  constexpr double toGiB = 1ull << 30;
  int K2 = K >> 1;
  int E = 9;

  int64_t dimX = 1ull << E;
  int64_t dimY = 1ull << E;
  int64_t dimZ = static_cast< int64_t >((memInGiB * toGiB) / (2.0/* in and output */ * sizeof(float) * dimY * dimX));
  int64_t N = dimZ * dimY * dimX;
  int64_t sizeInBytes = N * sizeof(float);
  int envSize = K * K * K;

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  int64_t offsetBytes = offsets.size() * sizeof(decltype(offsets)::value_type);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using block of dimensions " << dimX << " x " << dimY << " x " << dimZ << " (" << (2 * _N) << " " << unit << " overall); "
            << "Using regions of size " << K << " x " << K << " x " << K << ".\n";

  std::vector< float > data(N, 0);
  readData(data.data(), N);

  int pad = K - 1;
  int64_t sizeWithoutPadding = (dimX - pad) * (dimY - pad) * (dimZ - pad);
  std::vector< float > groundTruth(sizeWithoutPadding, 0);

  // Baseline: Single threaded and OpenMP-parallelized CPU implementation
  {
    out << memInGiB << ',';
    double cpuDuration;
    if (measureSingleThreadedCPU) {
      cpuDuration = cpuKernel(groundTruth.data(), data.data(), offsets.data(), N, K, dimX, dimY, dimZ, false);
      out << K << ',' << cpuDuration << ',';

    }
    cpuDuration = cpuKernel(groundTruth.data(), data.data(), offsets.data(), N, K, dimX, dimY, dimZ, true);
    out << K << ',' << cpuDuration;
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
    std::vector< float > host_output(groundTruth.size(), 0);

    // For testing the multi GPU scenario, we want as baseline some memory M processed on a single GPU.
    // In the multi GPU scenario, that memory should be split up on the devices, so each GPU processes only M/n_devices memory.
    // That split and the according launch on the multi GPU version is done within "launchKernelMultiCUDAGPU".

    auto call_cuda_kernel = [&](int threadsPerBlock) -> bool {
      float elapsedTimeInMilliseconds = -1;
      cudaError_t singleGPUError = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, 0, threadsPerBlock);
      if (singleGPUError != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]  Error during single GPU launch, error was '" << cudaGetErrorString(singleGPUError) << "'\n";
        return false;
      }

      out << ',' << threadsPerBlock << ',' << elapsedTimeInMilliseconds;

      cudaError_t cuda_kernel_error = launchKernelMultiCUDAGPU(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, threadsPerBlock);
      if (cuda_kernel_error != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]  Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'\n";
        return false;
      }

      out << ',' << elapsedTimeInMilliseconds;

      float maxCUDAError = computeMaxError(groundTruth.data(), host_output.data(), groundTruth.size());
      if (maxCUDAError > 1e-5) {
        std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
        return false;
      }
      return true;
    };

    int minThreadsPerBlock = std::min(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    int maxThreadsPerBlock = std::max(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    for (int threadsPerBlock = minThreadsPerBlock; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
      if (!call_cuda_kernel(1 << threadsPerBlock)) {
        goto AFTER_CUDA;
      }
    }

    int maxPotentialCUDABlockSize = 1;
    cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, 0/* multi GPU system: use the first device here only! */);
    if (blockSizeRetErr != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
    } else {
      call_cuda_kernel(maxPotentialCUDABlockSize);
    }
  }

AFTER_CUDA:;

#endif
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] << " <path/to/output/benchmark/csv>  <max-mem-in-GiB>  <num-percentages>  <Kmin>  <Kmax>  <threadsPerBlock>  <measure-single-threaded-CPU>\n\n";
    std::cerr << "  <path/to/output/benchmark/csv>  File path to the benchmark file that will be generated.\n";
    std::cerr << "  <max-mem-in-GiB>                Maximum memory to be used on GPU in GiB (float). If the device does not have sufficient memory, this will be reduced automatically.\n";
    std::cerr << "  <num-percentages>               Number of percentages of the max memory to sample. Must be bigger in [1..100], e.g. setting 2 will result in using 100% and 50%.\n";
    std::cerr << "  <Kmin>                          Minimum environment size, must be bigger than one and odd.\n";
    std::cerr << "  <Kmax>                          Maximum environment size, must be bigger than one and odd.\n";
    std::cerr << "  <threadsPerBlock>               Max (logarithmized) number of threads per block, e.g., 9 == 512 threads per block. The threads per block will get incremented by two each loop, i.e., [1, 4, ...]\n";
    std::cerr << "  <measure-single-threaded-CPU>   Indicator if the single threaded CPU execution shall be measured as well. Will be done if argument is 'y'.\n";
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

  std::string benchmarkLogFile  = argv[1];
  double maxMemoryGiB           = parseDouble(argv[2]);
  int numPercentages            = parseInt(argv[3]);
  int _Kmin                     = parseInt(argv[4]);
  int _Kmax                     = parseInt(argv[5]);
  int Kmin                      = std::min(_Kmin, _Kmax);
  int Kmax                      = std::max(_Kmin, _Kmax);
  int maxThreadsBlock           = parseInt(argv[6]);
  bool measureSingleThreadedCPU = argv[7][0] == 'y';

  maxMemoryGiB = getMaxAllocationSizeMultiCUDAGPU(maxMemoryGiB, 0.9);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryGiB << " GiB.\n";

  std::ofstream log(benchmarkLogFile);

  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    for (int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, percentage / 100.0 * maxMemoryGiB, K, maxThreadsBlock, measureSingleThreadedCPU);
      log << '\n';

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return EXIT_SUCCESS;
}