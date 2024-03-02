/*
 * MIT License
 * 
 * Copyright (c) 2024 Dr. Thomas Lang
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


void readData(float* buf, size_t N) {
  // Simulate some data reading process. This is meant to replace
  // any hard-coded values due to the memory mapping business below.
  std::fill_n(buf, N, 1.f);
}


void benchmarkMemoryTransfer(std::ostream& out, double memInGiB, int K, int numTransfers) {
  constexpr double toGiB = 1ull << 30;
  int K2 = K >> 1;
  int E = 9;

  int64_t dimX = 1ull << E;
  int64_t dimY = 1ull << E;
  int64_t dimZ = static_cast< int64_t >(memInGiB * toGiB / (2.0/* in and output */ * sizeof(float) * dimY * dimX));
  int64_t N = dimZ * dimY * dimX;
  int64_t sizeInBytes = N * sizeof(float);
  int envSize = static_cast< size_t >(K) * K * K;

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  size_t offsetBytes = offsets.size() * sizeof(int64_t);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using block of dimensions " << dimX << " x " << dimY << " x " << dimZ << " (" << (2 * _N) << " " << unit << " overall); "
            << "Using regions of size " << K << " x " << K << " x " << K << ".\n";

  int pad = K2 * 2;
  int64_t sizeWithoutPadding = (dimX - pad) * (dimY - pad) * (dimZ - pad);

  auto groundTruth = std::make_unique< float[] >(sizeWithoutPadding);
  std::memset(groundTruth.get(), 0, sizeWithoutPadding * sizeof(float));

  // Baseline: OpenMP-parallelized CPU implementation
  {
    auto data = std::make_unique< float[] >(N);
    readData(data.get(), N);
    out << memInGiB << ',';
    double cpuDuration = cpuKernel(groundTruth.get(), data.get(), offsets.data(), N, K, dimX, dimY, dimZ, true);
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
  }

  auto call_kernel = [&](float* inPtr, float* outPtr, int64_t* offsets) -> float {
    int maxPotentialCUDABlockSize = 1;
    cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, 0);
    if (blockSizeRetErr != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
      return -1;
    }

    float elapsedMs = -1;
    cudaError_t kernelError = launchKernel(outPtr, inPtr, N, offsets, K, dimX, dimY, dimZ, &elapsedMs, 0, maxPotentialCUDABlockSize);
    if (kernelError != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during kernel execution, error was '" << cudaGetErrorString(kernelError) << "'\n";
      return -1;
    }

    float maxCUDAError = computeMaxError(groundTruth.get(), outPtr, sizeWithoutPadding);
    if (maxCUDAError > 1e-5) {
      std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
      return -1;
    }

    return elapsedMs;
  };

  // Variant 1: Regular allocation, possible page-able memory.
  {
    auto pageableDataInput  = std::make_unique< float[] >(N);
    auto pageableDataOutput = std::make_unique< float[] >(sizeWithoutPadding);

    float avgElapsed = 0;
    for (int call = 0; call < numTransfers; ++call) {
      readData(pageableDataInput.get(), N);
      avgElapsed += call_kernel(pageableDataInput.get(), pageableDataOutput.get(), offsets.data());
    }

    out << ',' << avgElapsed / numTransfers;
  }

  // Variant 2: Pinned memory.
  {
    cudaError_t allocError = cudaError_t::cudaSuccess;
    float* pinnedDataInput, *pinnedDataOutput;
    int64_t* pinnedOffsets;
    allocError = cudaMallocHost(&pinnedDataInput,  N * sizeof(float));
    allocError = cudaMallocHost(&pinnedDataOutput, sizeWithoutPadding * sizeof(float));
    allocError = cudaMallocHost(&pinnedOffsets,    offsets.size() * sizeof(int64_t));
    if (allocError != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during host allocation, error was '" << cudaGetErrorString(allocError) << "'\n";
      return;
    }

    float avgElapsed = 0;
    for (int call = 0; call < numTransfers; ++call) {
      readData(pinnedDataInput, N);
      avgElapsed += call_kernel(pinnedDataInput, pinnedDataOutput, pinnedOffsets);
    }
 
    out << ',' << avgElapsed / numTransfers;

    (void)cudaFreeHost(pinnedDataInput);
    (void)cudaFreeHost(pinnedDataOutput);
    (void)cudaFreeHost(pinnedOffsets);
  }

  // Variant 3: Mapped memory.
  cudaDeviceProp prop;
  cudaError_t propError = cudaGetDeviceProperties(&prop, 0);
  if(error == cudaError_t::cudaSuccess && prop.canMapHostMemory)
  {
    float* mappedInput, *mappedOutput; 
    int64_t* mappedOffsets;
    cudaError_t mappingAllocError = cudaError_t::cudaSuccess;
    mappingAllocError = cudaHostAlloc(&mappedInput,   N * sizeof(float),                  cudaHostAllocMapped);
    mappingAllocError = cudaHostAlloc(&mappedOutput,  sizeWithoutPadding * sizeof(float), cudaHostAllocMapped);
    mappingAllocError = cudaHostAlloc(&mappedOffsets, offsets.size() * sizeof(int64_t),   cudaHostAllocMapped);
    if (mappingAllocError != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during mapped allocation, error was '" << cudaGetErrorString(mappingAllocError) << "'\n";
      return;
    }

    float* deviceInput, *deviceOutput;
    int64_t *deviceOffsets;
    cudaError_t devPtrError = cudaError_t::cudaSuccess;
    devPtrError = cudaHostGetDevicePointer(&deviceInput,   mappedInput,   0);
    devPtrError = cudaHostGetDevicePointer(&deviceOutput,  mappedOutput,  0);
    devPtrError = cudaHostGetDevicePointer(&deviceOffsets, mappedOffsets, 0);
    if (devPtrError != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during mapped device pointer retrieval, error was '" << cudaGetErrorString(devPtrError) << "'\n";
      return;
    }

    int maxPotentialCUDABlockSize = 1;
    cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, 0);
    if (blockSizeRetErr != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
      return;
    }

    cudaError_t kernelError = cudaError_t::cudaSuccess;
    float avgElapsed = 0;
    for (int call = 0; call < numTransfers; ++call) {
      readData(mappedInput, N);
      float elapsedMs;
      kernelError = launchKernelMapped(deviceOutput, deviceInput, N, deviceOffsets, K, dimX, dimY, dimZ, &elapsedMs, 0, maxPotentialCUDABlockSize);
      if (kernelError != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]   *** Error during kernel execution, error was '" << cudaGetErrorString(kernelError) << "'\n";
        return;
      }

      float maxCUDAError = computeMaxError(groundTruth.get(), mappedOutput, sizeWithoutPadding);
      if (maxCUDAError > 1e-5) {
        std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
        return;
      }

      avgElapsed += elapsedMs;
    }

    out << ',' << avgElapsed / numTransfers;

    (void)cudaFreeHost(mappedInput);
    (void)cudaFreeHost(mappedOutput);
    (void)cudaFreeHost(mappedOffsets);
  }

#endif
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " <path/to/output/benchmark/csv>  <max-mem-in-GiB>  <num-percentages>  <K>  <num-transfers>\n\n";
    std::cerr << "  <path/to/output/benchmark/csv>  File path to the benchmark file that will be generated.\n";
    std::cerr << "  <max-mem-in-GiB>                Maximum memory to be used on GPU in GiB (float). If the device does not have sufficient memory, this will be reduced automatically.\n";
    std::cerr << "  <num-percentages>               Number of percentages of the max memory to sample. Must be bigger in [1..100], e.g. setting 2 will result in using 100% and 50%.\n";
    std::cerr << "  <K>                             Environment size, must be bigger than one and odd.\n";
    std::cerr << "  <num-transfers>                 Number of memory transfers of once allocated memory.\n";
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
  int K                         = parseInt(argv[4]);
  int numTransfers              = parseInt(argv[5]);

  double maxMemoryInGB = getMaxMemoryInGiB(maxMemoryGiB, 0.9, false);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n";


  std::ofstream log(benchmarkLogFile);

  bool firstRun = true;
  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    benchmarkMemoryTransfer(log, percentage / 100.0 * maxMemoryInGB, K, numTransfers);
    log << '\n';
  }

  return EXIT_SUCCESS;
}