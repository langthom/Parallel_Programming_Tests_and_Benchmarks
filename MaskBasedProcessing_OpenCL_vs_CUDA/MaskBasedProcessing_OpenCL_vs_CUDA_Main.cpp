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

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../Common/CommonFunctions.h"
#include "../Common/CommonKernels.h"


void readData(float* buf, int64_t N) {
  #pragma omp parallel for
  for(int64_t i = 0; i < N; ++i) {
    buf[i] = i % 65536;
  }
}

constexpr int minLogThreadsPerBlock() {
  // Yield the minimum (logarithmized) number of threads per block.
  // Since we experienced time-outs if we only use 1 thread per block,
  // we use a minimum of 2**4 == 16 threads per block.
  return 4;
}

std::tuple< int64_t, int64_t, int64_t > getDims(double memInGiB, int alignmentZ = 0) {
  constexpr double toGiB = 1ull << 30;
  int64_t dimX = 1ull << 9;
  int64_t dimY = 1ull << 9;
  int64_t dimZ = static_cast< int64_t >(memInGiB * toGiB / (2.0/* in and output */ * sizeof(float) * dimY * dimX));

  if (alignmentZ > 0) {
    dimZ -= dimZ % (1ll << alignmentZ);
  }

  return { dimX, dimY, dimZ };
}

int correctMaxThreadsPerBlock(double memInGiB, int maxLogThreadsPerBlock) {
  // Given a user provided maximum number of threads per block, this might yield alignment issues
  // with OpenCL as this requires the global workgroup size (here: dimZ * dimY * dimX) to be 
  // evenly divisible by the local range (here: the number of threads per block).
  // Since we also calculate the maximum usable memory, we must NOT pad up.
  // Instead, we reduce the maximum number of threads per block until the overall number of elements is
  // evenly divisible by it and report this to the user.

  auto dims = getDims(memInGiB);
  int64_t dimZ = std::get< 2 >(dims);

  for (; maxLogThreadsPerBlock >= minLogThreadsPerBlock(); --maxLogThreadsPerBlock) {
    int64_t threadsPerBlock = 1ll << maxLogThreadsPerBlock;
    int64_t roundDown = dimZ - (dimZ & (threadsPerBlock - 1ll));
    if (roundDown != 0) {
      break;
    }
  }

  return maxLogThreadsPerBlock;
}

void benchmark(std::ostream& out, double memInGiB, int K, bool printSpecs, int inMaxThreadsPerBlock, bool measureSingleThreaded) {
  int K2 = K >> 1;

  int64_t dimX, dimY, dimZ;
  std::tie(dimX, dimY, dimZ) = getDims(memInGiB, inMaxThreadsPerBlock);

  int64_t N = dimZ * dimY * dimX;
  int64_t sizeInBytes = N * sizeof(float);
  int envSize = K * K * K;

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  int64_t offsetBytes = offsets.size() * sizeof(int64_t);

  if(printSpecs) {
#ifdef HAS_CUDA
    int nr_gpus = 0;
    std::vector< std::string > deviceNames;
    std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
    cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

    if (error != cudaError_t::cudaSuccess) {
      std::cout << "[CUDA] Error during GPU information retrieval, error was: " << cudaGetErrorString(error);
      return;
    } else {
      std::cout << "[CUDA] Detected " << nr_gpus << " GPUs that can run CUDA.\n";
      for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
        std::string freeUnit, totalUnit;
        float freeMem  = formatMemory(availableMemoryPerDevice[gpuID], freeUnit);
        float totalMem = formatMemory(totalMemoryPerDevice[gpuID], totalUnit);
        float frac = (double)availableMemoryPerDevice[gpuID] / (double)totalMemoryPerDevice[gpuID];
        std::cout << "[CUDA]   Device " << gpuID << " (" << deviceNames[gpuID] << ") has "
          << std::setprecision(3) << freeMem << ' ' << freeUnit << " free memory out of "
          << std::setprecision(3) << totalMem << ' ' << totalUnit
          << " (-> " << std::setprecision(4) << frac * 100.f << " %)\n";

        int maxPotentialBlockSize;
        error = getMaxPotentialBlockSize(maxPotentialBlockSize, gpuID);
        if(error != cudaError_t::cudaSuccess) {
          std::cerr << "[CUDA]    Error while retrieving the max. potential block size, error was: " << cudaGetErrorString(error) << '\n';
        } else {
          std::cout << "[CUDA]    - max potential block size for device " << gpuID << " is: " << maxPotentialBlockSize << '\n';
        }
      }
    }
#endif // HAS_CUDA

#ifdef HAS_OPENCL
    std::vector< cl::Platform > platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
      std::cerr << "[ CL ] No OpenCL platforms could be detected!\n";
    } else {
      std::cout << "[ CL ] Detected " << platforms.size() << " OpenCL platform(s).\n";
      for (auto const& platform : platforms) {
        std::cout << "[ CL ]   '" << platform.getInfo< CL_PLATFORM_NAME >() << "'\n";

        std::vector< cl::Device > devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (!devices.empty()) {
          std::cout << "[ CL ]     Detected " << devices.size() << " device(s) on this platform:\n";
          for (auto const& device : devices) {
            std::string constMemUnit;
            float constMem = formatMemory(device.getInfo< CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE >(), constMemUnit);
            std::string globalMemUnit;
            float globalMem = formatMemory(device.getInfo< CL_DEVICE_GLOBAL_MEM_SIZE >(), globalMemUnit);
            std::string localMemUnit;
            float localMem = formatMemory(device.getInfo< CL_DEVICE_LOCAL_MEM_SIZE >(), localMemUnit);

            bool supportsImages = device.getInfo< CL_DEVICE_IMAGE_SUPPORT >();

            std::cout << "[ CL ]       - Device name:            '" << device.getInfo< CL_DEVICE_NAME >() << "'\n";
            std::cout << "[ CL ]       - Device global   memory: " << std::setw(3) << std::setprecision(3) << globalMem << ' ' << globalMemUnit << '\n';
            std::cout << "[ CL ]       - Device constant memory: " << std::setw(3) << std::setprecision(3) << constMem << ' ' << constMemUnit << '\n';
            std::cout << "[ CL ]       - Device local    memory: " << std::setw(3) << std::setprecision(3) << localMem << ' ' << localMemUnit << '\n';
            std::cout << "[ CL ]       - Device supports images? " << std::boolalpha << supportsImages;
            if (supportsImages) {
              std::cout << " (max. dimensions: "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_WIDTH  >() << " x "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_HEIGHT >() << " x "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_DEPTH  >() << ")";
            }
            std::cout << "\n";
            std::cout << "[ CL ]       - Max workgroup size:     " << device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >() << '\n';
          }
        }
      }
    }
#endif // HAS_OPENCL

#if defined(HAS_CUDA) || defined(HAS_OPENCL)
    std::cout << '\n';
#endif
  }

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using block of dimensions " << dimX << " x " << dimY << " x " << dimZ << " (" << (2 * _N) << " " << unit << " overall); "
            << "Using regions of size " << K << " x " << K << " x " << K << ".\n";

  int pad = K2 * 2;
  int64_t sizeWithoutPadding = (dimX - pad) * (dimY - pad) * (dimZ - pad);
  std::vector< float > groundTruth(sizeWithoutPadding, 0);

  // Baseline: Single threaded and OpenMP-parallelized CPU implementation
  {
    std::vector< float > data(N, 0);
    readData(data.data(), N);
    out << memInGiB << ',';
    double cpuDuration;
    if (measureSingleThreaded) {
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
    std::cerr << "[CUDA] Error during GPU information retrieval, error was: " << cudaGetErrorString(error);
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

      float maxCUDAError = computeMaxError(groundTruth.data(), host_output.data(), groundTruth.size());
      if (maxCUDAError > 1e-5) {
        std::cerr << "[CUDA]   *** Max error with CUDA execution is: " << maxCUDAError << '\n';
        return false;
      }
      return true;
    };

    int minThreadsPerBlock = std::min(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    int maxThreadsPerBlock = std::max(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      for(int threadsPerBlock = minThreadsPerBlock; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
        call_cuda_kernel(gpuID, 1ull << threadsPerBlock);
      }

      int maxPotentialCUDABlockSize = 1;
      cudaError_t blockSizeRetErr = getMaxPotentialBlockSize(maxPotentialCUDABlockSize, gpuID);
      if(blockSizeRetErr != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]   *** Error during retrieval of maximum potential block size: " << cudaGetErrorString(blockSizeRetErr) << '\n';
      } else {
        call_cuda_kernel(gpuID, maxPotentialCUDABlockSize);
      }
    }
  }
#endif // HAS_CUDA

  // Call the OpenCL kernel
#ifdef HAS_OPENCL

#define HANDLE_OCL_ERROR(line) if (error != CL_SUCCESS) { std::cerr << "[CL]  *** Error: " << getOpenCLError(error, line) << '\n'; return false; }

  auto callOpenCLKernel = [&](cl::Device const& device, int threadsPerBlock) -> bool {
    // Initialize the OpenCL stuff. This is specific to each device, thus we need to repeat this for every device.
    cl_command_queue_properties queueProperties = CL_QUEUE_PROFILING_ENABLE;
    cl::Context context(device);
    cl::CommandQueue queue(context, device, queueProperties);

    cl::NDRange globalRange(N);
    cl::NDRange localRange(threadsPerBlock);

    auto kernelCode = getOpenCLKernel();
    cl::Program::Sources sources;
    sources.push_back({ kernelCode.c_str(), kernelCode.length() });

    cl::Program program(context, sources);
    if (program.build({ device }) != CL_SUCCESS) {
      std::cerr << "[CL]  *** Error while building OpenCL code: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device);
      return false;
    }
    cl_int error = CL_SUCCESS;

    cl::Buffer deviceIn( context, CL_MEM_READ_WRITE, sizeInBytes,                        nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);
    cl::Buffer deviceOut(context, CL_MEM_READ_WRITE, sizeWithoutPadding * sizeof(float), nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);
    cl::Buffer deviceOff(context, CL_MEM_READ_ONLY,  offsetBytes, nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);

    // Prepare the kernel.
    cl::Kernel kernel(program, "useStats");
    error |= kernel.setArg(0, deviceIn);
    error |= kernel.setArg(1, deviceOut);
    error |= kernel.setArg(2, deviceOff);
    error |= kernel.setArg(3, K);
    error |= kernel.setArg(4, N);
    error |= kernel.setArg(5, dimX);
    error |= kernel.setArg(6, dimY);
    error |= kernel.setArg(7, dimZ);
    HANDLE_OCL_ERROR(__LINE__);


    std::vector< float > host_output(groundTruth.size(), 0);
    std::vector< float > data(N, 0);
    readData(data.data(), N);

    // The timing start for the execution of the memory transfer and the kernel execution.
    // As the data reading operation is expensive, we assume copying memory already in host RAM
    // to the device, but now by means of memory mapping. This means more memory consumption than
    // directly reading the data to the memory mapped block, but this would imply a non-fair 
    // comparison between the CUDA part and the OpenCL part.
    // 
    // Furthermore, since we know use the regular enqueueWriteBuffer approach, we want to make the
    // comparison fairer by placing these calls asynchronously (!). The kernel launch itself should
    // also run asynchronously. The final enqueueReadBuffer command is blocking, which does the final
    // synchronization and allows for a CPU timing instead of using different events.
    // 
    // Note that we use CPU code here since the blocks below are all *blocking*.
    auto oclExecStart = std::chrono::high_resolution_clock::now();

    // Write data from host to device (asynchronous).
    error |= queue.enqueueWriteBuffer(deviceIn,  CL_FALSE, 0, sizeInBytes, data.data());
    error |= queue.enqueueWriteBuffer(deviceOff, CL_FALSE, 0, offsetBytes, offsets.data());
    HANDLE_OCL_ERROR(__LINE__);

    // Call the kernel actually.
    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
    HANDLE_OCL_ERROR(__LINE__);

    // Read data from device to host (blocking!).
    error |= queue.enqueueReadBuffer(deviceOut, CL_TRUE, 0, sizeWithoutPadding * sizeof(float), host_output.data());
    HANDLE_OCL_ERROR(__LINE__);

    // Profiling information:
    auto oclExecEnd = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration< double, std::milli >(oclExecEnd - oclExecStart).count();
    out << ',' << threadsPerBlock << ',' << elapsedMs;

    float maxOpenCLError = computeMaxError(groundTruth.data(), host_output.data(), sizeWithoutPadding);
    if (maxOpenCLError > 1e-5) {
      std::cerr << "[CL] Max error with OpenCL execution is: " << maxOpenCLError << '\n';
      return false;
    }

    return true;
  };

  std::vector< cl::Platform > platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "[ CL ] No OpenCL platforms could be detected!\n";
  } else {
    int minThreadsPerBlock = std::min(minLogThreadsPerBlock(), inMaxThreadsPerBlock);
    int maxThreadsPerBlock = std::max(minLogThreadsPerBlock(), inMaxThreadsPerBlock);

    for (auto const& platform : platforms) {
      std::vector< cl::Device > devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

      if (!devices.empty()) {
        for (auto const& device : devices) {
          for(int threadsPerBlock = minThreadsPerBlock; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
            callOpenCLKernel(device, 1ull << threadsPerBlock);
          }

          // For the OpenCL invocation, we do NOT test the maximum potential CUDA block size.
          // The reason is, that the function "enqueueNDRangeKernel" requires that the global work group
          // size (in our case the number of elements) is evenly divisible (i.e., without remainder) by
          // the local group size. For the powers of two in the loop above, this should work.
          // However, the max. potential CUDA block size need not be a power of two. In one of our tested cases,
          // it was 768, and the global size was not evenly divisible by that.
          // Therefore, we do not test this and ignore this.
        }
      }
    }
  }
#undef HANDLE_OCL_ERROR

#endif // HAS_OPENCL
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

  std::string benchmarkLogFile = argv[1];
  double maxMemoryGiB           = parseDouble(argv[2]);
  int numPercentages            = parseInt(argv[3]);
  int _Kmin                     = parseInt(argv[4]);
  int _Kmax                     = parseInt(argv[5]);
  int Kmin                      = std::min(_Kmin, _Kmax);
  int Kmax                      = std::max(_Kmin, _Kmax);
  int maxThreadsBlock           = parseInt(argv[6]);
  bool measureSingleThreadedCPU = argv[7][0] == 'y';

  double maxMemoryInGB = getMaxMemoryInGiB(maxMemoryGiB);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n";

  std::ofstream log(benchmarkLogFile);

  bool firstRun = true;
  int percentageStep = 100 / numPercentages;

  double minMemory = maxMemoryInGB * (100. - static_cast< double >(numPercentages-1.0) * percentageStep) / 100.;
  int _oldMaxThreads = maxThreadsBlock;
  maxThreadsBlock = correctMaxThreadsPerBlock(minMemory, maxThreadsBlock);
  if (_oldMaxThreads != maxThreadsBlock) {
    std::cout << "[INFO]  Correction: Using " << maxThreadsBlock << " threads/block at max instead of " << _oldMaxThreads << " for OpenCL alignment reasons\n";
  }
  std::cout << '\n';

  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    for(int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, percentage / 100.0 * maxMemoryInGB, K, firstRun, maxThreadsBlock, measureSingleThreadedCPU);
      log << '\n';
      firstRun = false;

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return EXIT_SUCCESS;
}