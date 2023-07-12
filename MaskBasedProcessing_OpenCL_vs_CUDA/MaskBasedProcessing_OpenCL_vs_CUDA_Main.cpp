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

inline std::array< float, 4 > stats(float* buffer, int* offsets, int index, int K) {
  int envSize = K * K * K;
  std::vector< float > env(envSize);
  for (int j = 0; j < envSize; ++j) {
    env[j] = buffer[index + offsets[j]];
  }

  std::array< float, 4 > features;

  float sum = 0.f;

  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += env[voxelIndex];
  }
  float mean = sum / envSize;

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = env[voxelIndex] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float stdev = std::sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = env[voxelIndex] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
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

void readData(float* buf, size_t N) {
  #pragma omp parallel for
  for(int i = 0; i < static_cast< int >(N); ++i) {
    buf[i] = i % 65536;
  }
}

constexpr int minLogThreadsPerBlock() {
  // Yield the minimum (logarithmized) number of threads per block.
  // Since we experienced time-outs if we only use 1 thread per block,
  // we use a minimum of 2**4 == 16 threads per block.
  return 4;
}

void benchmark(std::ostream& out, double memInGiB, int K, bool printSpecs, int inMaxThreadsPerBlock) {
  int K2 = K >> 1;
  int E = 9;

  size_t dimX = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimZ = (size_t)(memInGiB * (1ull << 30)) / (2/* in and output */ * sizeof(float) * dimY * dimX);
  size_t N = dimZ * dimY * dimX;
  size_t sizeInBytes = N * sizeof(float);
  size_t envSize = static_cast< size_t >(K) * K * K;

  auto offsets = computeMaskOffsets(K, dimY, dimX);
  size_t offsetBytes = offsets.size() * sizeof(int);

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
  std::cout << "Using " << N << " elements, corresponding to " << (2*_N) << " " << unit << " overall; Using regions of size " << K << " x " << K << " x " << K << ".\n";

  std::vector< float > groundTruth(N, 0);

  // Baseline: OpenMP-parallelized implementation
  {
    std::vector< float > data(N, 0);
    readData(data.data(), N);
    float* dataPtr = data.data();
    auto cpuStart = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
      size_t posZ = i / (dimX * dimY);
      size_t it   = i - posZ * dimY * dimX;
      size_t posY = it / dimX;
      size_t posX = it % dimX;

      if (posX < K2 || posY < K2 || posZ < K2 || posX > dimX - 1 - K2 || posY > dimY - 1 - K2 || posZ > dimZ - 1 - K2) {
        groundTruth[i] = 0.f;
      }
      else {
        auto features = stats(dataPtr, offsets.data(), i, K);
        groundTruth[i] = features[0];
      }
    }

    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast< std::chrono::milliseconds >(cpuEnd - cpuStart).count();
    out << memInGiB << ',' << K << ',' << cpuDuration;

  }

  auto computeMaxError = [&groundTruth,N](float* host_output) -> float {
    float maxError = 0.f;
    float* outIt = host_output;
    float* inIt  = groundTruth.data();

    #pragma omp parallel
    {
      float maxErrorForThisThread = 0.f;

      #pragma omp for nowait
      for(int i = 0; i < N; ++i) {
        float absError = std::fabsf(inIt[i] - outIt[i]);
        maxErrorForThisThread = std::fmaxf(maxErrorForThisThread, absError);
      }

      #pragma omp critical
      maxError = std::fmaxf(maxError, maxErrorForThisThread);
    }

    return maxError;
  };

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
    std::vector< float > host_output(N, 0);
    readData(data.data(), N);

    auto call_cuda_kernel = [&](int deviceID, int threadsPerBlock) -> bool {
      float elapsedTimeInMilliseconds = -1;
      cudaError_t cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, deviceID, threadsPerBlock);
      if (cuda_kernel_error != cudaError_t::cudaSuccess) {
        std::cerr << "[CUDA]  Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'\n";
        return false;
      }

      out << ',' << threadsPerBlock << ',' << elapsedTimeInMilliseconds;

      float maxCUDAError = computeMaxError(host_output.data());
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
        if(!call_cuda_kernel(gpuID, 1ull << threadsPerBlock)) {
          goto AFTER_CUDA;
        }
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

AFTER_CUDA:;

#endif

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

    // See the large comment in "MeasureMaxUsableMemory_Main.cpp" for a description of the following mapped-memory code.

    cl::Buffer deviceIn( context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeInBytes, nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);
    cl::Buffer deviceOut(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeInBytes, nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);
    cl::Buffer deviceOff(context, CL_MEM_READ_ONLY,  offsetBytes, nullptr, &error);
    HANDLE_OCL_ERROR(__LINE__);

    {
      float* input_mapped = static_cast< float* >(queue.enqueueMapBuffer(deviceIn, CL_TRUE, CL_MAP_WRITE, 0, sizeInBytes, nullptr, nullptr, &error));
      HANDLE_OCL_ERROR(__LINE__);
      readData(input_mapped, N);
      error |= queue.enqueueUnmapMemObject(deviceIn, input_mapped);
      HANDLE_OCL_ERROR(__LINE__);
    }

    error |= queue.enqueueWriteBuffer(deviceOff, CL_TRUE, 0, offsetBytes, offsets.data());
    HANDLE_OCL_ERROR(__LINE__);

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

    cl::Event profileEvent;

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange, nullptr, &profileEvent);
    HANDLE_OCL_ERROR(__LINE__);

    error |= queue.finish();
    HANDLE_OCL_ERROR(__LINE__);

    {
      float* host_output_mapped = static_cast< float* >(queue.enqueueMapBuffer(deviceOut, CL_TRUE, CL_MAP_READ, 0, sizeInBytes, nullptr, nullptr, &error));
      HANDLE_OCL_ERROR(__LINE__);

      // Profiling information:
      cl_ulong timeStart = profileEvent.getProfilingInfo< CL_PROFILING_COMMAND_START >(&error);
      cl_ulong timeEnd   = profileEvent.getProfilingInfo< CL_PROFILING_COMMAND_END   >(&error);
      HANDLE_OCL_ERROR(__LINE__);

      long double elapsed = static_cast<long double>(timeEnd - timeStart);
      long double elapsedMs = elapsed * 1e-6;
      out << ',' << threadsPerBlock << ',' << elapsedMs;

      float maxOpenCLError = computeMaxError(host_output_mapped);
      if (maxOpenCLError > 1e-5) {
        std::cerr << "[CL] Max error with OpenCL execution is: " << maxOpenCLError << '\n';
        return false;
      }

      error |= queue.enqueueUnmapMemObject(deviceOut, host_output_mapped);
      HANDLE_OCL_ERROR(__LINE__);
    }

#undef HANDLE_OCL_ERROR
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
            if (!callOpenCLKernel(device, 1ull << threadsPerBlock)) {
              goto AFTER_OPENCL;
            }
          }

          // For the OpenCL invocation, we do NOT test the maximum potential CUDA block size.
          // The reason is, that the function "enqueueNDRangeKernel" requires that the global work group
          // size (in our case the number of elements) is evenly divisible (i.e., without remainder) by
          // the local group size. For the powers of two in the loop above, this should work.
          // However, the max. potential CUDA block size need not be a power of two. In our tested case,
          // it was 768, and the global size was not evenly divisible.
          // Therefore, we do not test this and ignore this.

          AFTER_OPENCL:;
        }
      }
    }
  }

  out << '\n';

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
  int numPercentages  = parseInt(argv[3]);
  int _Kmin           = parseInt(argv[4]);
  int _Kmax           = parseInt(argv[5]);
  int Kmin            = std::min(_Kmin, _Kmax);
  int Kmax            = std::max(_Kmin, _Kmax);
  int maxThreadsBlock = parseInt(argv[6]);


  double maxMemoryInGB = getMaxMemoryInGiB(maxMemoryGiB);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n";

  std::ofstream log(benchmarkLogFile);

  bool firstRun = true;
  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    for(int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, percentage / 100.0 * maxMemoryInGB, K, firstRun, maxThreadsBlock);
      log << std::string(100, '=') << '\n';
      firstRun = false;

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  return EXIT_SUCCESS;
}