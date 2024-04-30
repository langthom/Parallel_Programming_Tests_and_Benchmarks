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
#include <random>
#include <sstream>
#include <thread>
#include <tuple>

#include "../Common/CommonFunctions.h"
#include "../Common/CommonKernels.h"
#include "NDLaunchTest.h"

void readData(float* buf, size_t N) {
  std::random_device rd;
  std::default_random_engine re(rd());
  std::uniform_real_distribution<float> dist(0, 65535);
  std::for_each(buf, buf + N, [&dist,&re](float& f){ f = dist(re); });
}

std::tuple< int_least64_t, int_least64_t, int_least64_t > GetDims(double memInGiB) {
  constexpr double toGiB = 1ull << 30;
  int E = 10;
  int_least64_t dimX = 1ull << E;
  int_least64_t dimY = 1ull << E;
  int_least64_t dimZ = static_cast< int_least64_t >(memInGiB * toGiB / (2.0/* in and output */ * sizeof(float) * dimY * dimX));
  return std::make_tuple(dimX, dimY, dimZ);
}

std::unique_ptr< float[] > GetData(double memInGiB) {
  auto const [dimX, dimY, dimZ] = GetDims(memInGiB);
  int_least64_t N = dimZ * dimY * dimX;
  auto data = std::make_unique< float[] >(N);
  readData(data.get(), N);
  return data;
}

void benchmark(std::ostream& out, double memInGiB, std::unique_ptr< float[] > const& data, int K) {
  int_least64_t dimX, dimY, dimZ;
  std::tie(dimX, dimY, dimZ) = GetDims(memInGiB);

  int K2 = K >> 1;
  int_least64_t N = dimZ * dimY * dimX;
  int_least64_t sizeInBytes = N * sizeof(float);

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  int_least64_t offsetBytes = offsets.size() * sizeof(int_least64_t);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using block of dimensions " << dimX << " x " << dimY << " x " << dimZ << " (" << (2 * _N) << " " << unit << " overall); "
            << "Using regions of size " << K << " x " << K << " x " << K << ".\n";

  int pad = K2 * 2;
  int_least64_t sizeWithoutPadding = (dimX - pad) * (dimY - pad) * (dimZ - pad) * sizeof(float);

  auto groundTruth = std::make_unique< float[] >(sizeWithoutPadding);
  std::memset(groundTruth.get(), 0, sizeWithoutPadding * sizeof(float));

  // OpenMP for a ground truth solution
  {
    out << memInGiB << ',';
    double cpuDuration;
    cpuDuration = cpuKernel(groundTruth.get(), data.get(), offsets.data(), N, K, dimX, dimY, dimZ, true);
    out << K << ',' << cpuDuration;
  }

  // Call the CUDA kernel
#ifdef HAS_CUDA
  auto host_output = std::make_unique< float[] >(sizeWithoutPadding);
  std::memset(host_output.get(), 0, sizeWithoutPadding * sizeof(float));

  auto callCudaKernel = [&](dim3 threads, int dim) -> bool {
    float elapsedMilliseconds = -1;
    cudaError_t launchError = launchKernelND(host_output.get(), data.get(), N, offsets.data(), K, dimX, dimY, dimZ, threads, &elapsedMilliseconds, /*deviceID=*/0);
    if(launchError != cudaError_t::cudaSuccess) {
      std::cerr << "[CUDA]  Error during " << dim << "D kernel launch, error was '" << cudaGetErrorString(launchError) << "'\n";
      return false;
    }

    float maxCUDAError = computeMaxError(groundTruth.get(), host_output.get(), sizeWithoutPadding);
    if (maxCUDAError > 1e-5) {
      std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
      return false;
    }

    out << ',' << elapsedMilliseconds;
    return true;
  };

  std::array< dim3, 3 > blocks{
    dim3(1024,1,1), dim3(32,32,1), dim3(16,16,4)
  };

  for(int dim = 0; dim < blocks.size(); ++dim) {
    bool success = callCudaKernel(blocks.at(dim), dim);
    out.flush();
    if(!success) {
      return;
    }
  }

#endif
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " <path/to/output/benchmark/csv>  <max-mem-in-GiB>  <num-percentages>  <Kmin>  <Kmax> \n\n";
    std::cerr << "  <path/to/output/benchmark/csv>  File path to the benchmark file that will be generated.\n";
    std::cerr << "  <max-mem-in-GiB>                Maximum memory to be used on GPU in GiB (float). If the device does not have sufficient memory, this will be reduced automatically.\n";
    std::cerr << "  <num-percentages>               Number of percentages of the max memory to sample. Must be bigger in [1..100], e.g. setting 2 will result in using 100% and 50%.\n";
    std::cerr << "  <Kmin>                          Minimum environment size, must be bigger than one and odd.\n";
    std::cerr << "  <Kmax>                          Maximum environment size, must be bigger than one and odd.\n";
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

  int deviceID = 0;
  {
    int nr_gpus;
    std::vector< std::string > deviceNames;
    std::vector< std::size_t > availableMemoryPerDevice, totalMemoryPerDevice;
    cudaError_t _err = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

    double avlMem, totMem;
    std::string avlMemUnit, totMemUnit;
    avlMem = formatMemory(availableMemoryPerDevice.at(deviceID), avlMemUnit);
    totMem = formatMemory(    totalMemoryPerDevice.at(deviceID), totMemUnit);

    std::cout << "[INFO]  Device #" << deviceID << ":\n";
    std::cout << "[INFO]     Name:  " << deviceNames.at(deviceID) << "\n";
    std::cout << "[INFO]     Available memory: " << avlMem << " [" << avlMemUnit << "]\n";
    std::cout << "[INFO]     Total     memory: " << totMem << " [" << totMemUnit << "]\n";
    std::cout << "\n";
  }

  double maxMemoryInGB = getMaxMemoryInGiB(maxMemoryGiB, 0.9, false);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n\n";

  std::ofstream log(benchmarkLogFile);
  log << "Mem [GiB],K,CPU [ms],GPU 1D [ms],GPU 2D [ms], GPU 3D [ms]\n";

  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    double currentMemoryGiB = percentage / 100.0 * maxMemoryInGB;
    auto data = GetData(currentMemoryGiB);

    for (int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, currentMemoryGiB, data, K);
      log << '\n';

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return EXIT_SUCCESS;
}