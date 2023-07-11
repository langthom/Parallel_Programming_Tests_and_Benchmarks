
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
#include <vector>

#ifdef HAS_CUDA
#include "../CommonKernels/CommonKernels.h"
#endif

#ifdef HAS_OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>
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

void benchmark(std::ostream& out, double memInGB, int K, bool printSpecs, int maxThreadsPerBlock) {
  int K2 = K >> 1;
  int E = 9;

  size_t dimX = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimZ = (size_t)(memInGB * (1ull << 30)) / (sizeof(float) * dimY * dimX);
  size_t N = dimZ * dimY * dimX;
  size_t sizeInBytes = N * sizeof(float);

  std::string unit;
  float _N = formatMemory(sizeInBytes, unit);
  std::cout << "Using " << N << " elements, corresponding to " << _N << " " << unit << "; Using regions of size " << K << " x " << K << " x " << K << ".\n";

  // Prepare the random input data.
  std::random_device rd;
  std::mt19937 gen{ rd() };
  std::uniform_real_distribution< float > dist{ 0.f, 100.f };

  std::vector< float > data(N, 0);
  std::vector< float > host_output(N, 0);
  std::for_each(data.begin(), data.end(), [&](float& f) {f = dist(gen); });

  // Prepare the ground truth results on CPU.
  size_t envSize = static_cast<size_t>(K) * K * K;
  std::vector< size_t > offsets(envSize, 0);
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
    size_t it = i - posZ * dimY * dimX;
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
  out << memInGB << ',' << K << ',' << cpuDuration;

  // Call the CUDA kernel
#ifdef HAS_CUDA

  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  int maxPotentialCUDABlockSizeDev0 = 1;

  if (error != cudaError_t::cudaSuccess) {
    std::cerr << "Error during GPU information retrieval, error was: " << cudaGetErrorString(error) << '\n';
  } else {

    if (printSpecs) {
      std::cout << "[INFO] Detected " << nr_gpus << " GPUs that can run CUDA.\n";
      for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
        std::string freeUnit, totalUnit;
        float freeMem  = formatMemory(availableMemoryPerDevice[gpuID], freeUnit);
        float totalMem = formatMemory(totalMemoryPerDevice[gpuID], totalUnit);
        float frac = (double)availableMemoryPerDevice[gpuID] / (double)totalMemoryPerDevice[gpuID];
        std::cout << "[INFO]   Device " << gpuID << " (" << deviceNames[gpuID] << ") has "
          << std::setprecision(3) << freeMem << ' ' << freeUnit << " free memory out of "
          << std::setprecision(3) << totalMem << ' ' << totalUnit
          << " (-> " << std::setprecision(4) << frac * 100.f << " %)\n";

        int maxPotentialBlockSize;
        error = getMaxPotentialBlockSize(maxPotentialBlockSize, gpuID);
        if(error != cudaError_t::cudaSuccess) {
          std::cerr << "Error while retrieving the max. potential block size, error was: " << cudaGetErrorString(error) << '\n';
        } else {
          std::wcout << "[INFO]    - max potential block size for device " << gpuID << " is: " << maxPotentialBlockSize << '\n';
        }
      }
    }

    auto call_cuda_kernel = [&](int deviceID, int threadsPerBlock) -> bool {
      float elapsedTimeInMilliseconds = -1;
      cudaError_t cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, deviceID, threadsPerBlock);
      if (cuda_kernel_error != cudaError_t::cudaSuccess) {
        std::cerr << "Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'\n";
        return false;
      }

      out << ',' << threadsPerBlock << ',' << elapsedTimeInMilliseconds;

      float maxCUDAError = 0.f;
      auto inIt = groundTruth.begin(), outIt = host_output.begin();
      for (; inIt != groundTruth.end(); ++inIt, ++outIt) {
        maxCUDAError = std::fmaxf(maxCUDAError, std::fabsf(*inIt - *outIt));
      }
      if (maxCUDAError > 1e-5) {
        std::cerr << "Max error with CUDA execution is: " << maxCUDAError << '\n';
      }
      return true;
    };

    for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      for(int threadsPerBlock = 0; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
        if(!call_cuda_kernel(gpuID, 1ull << threadsPerBlock)) {
          goto AFTER_CUDA;
        }
      }

      if(gpuID == 0) {
        getMaxPotentialBlockSize(maxPotentialCUDABlockSizeDev0, gpuID);
      }

      call_cuda_kernel(gpuID, maxPotentialCUDABlockSizeDev0);
    }
  }

AFTER_CUDA:;

#endif

  // Call the OpenCL kernel
#ifdef HAS_OPENCL

  std::for_each(host_output.begin(), host_output.end(), [](float& o) { o = 0; }); // Reset the contents of the buffer that shall contain the results.

  auto callOpenCLKernel = [&](cl::Device const& device, int threadsPerBlock) -> bool {
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
      std::cerr << "Error while building OpenCL code: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device) << '\n';
      return false;
    }
    cl_int error = CL_SUCCESS;

    cl::Buffer deviceIn(context, CL_MEM_READ_ONLY, sizeInBytes);
    cl::Buffer deviceOut(context, CL_MEM_READ_WRITE, sizeInBytes);
    cl::Buffer deviceOff(context, CL_MEM_READ_ONLY, envSize * sizeof(size_t));

    error |= queue.enqueueWriteBuffer(deviceIn, CL_TRUE, 0, sizeInBytes, data.data());
    if (error != CL_SUCCESS) return false;

    error |= queue.enqueueWriteBuffer(deviceOff, CL_TRUE, 0, envSize * sizeof(size_t), offsets.data());
    if (error != CL_SUCCESS) return false;

    cl::Kernel kernel(program, "useStats");
    error |= kernel.setArg(0, deviceIn);
    error |= kernel.setArg(1, deviceOut);
    error |= kernel.setArg(2, deviceOff);
    error |= kernel.setArg(3, K);
    error |= kernel.setArg(4, N);
    error |= kernel.setArg(5, dimX);
    error |= kernel.setArg(6, dimY);
    error |= kernel.setArg(7, dimZ);
    if (error != CL_SUCCESS) return false;

    cl::Event profileEvent;

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange, nullptr, &profileEvent);
    if (error != CL_SUCCESS) return false;

    error |= queue.enqueueReadBuffer(deviceOut, CL_TRUE, 0, sizeInBytes, host_output.data());
    if (error != CL_SUCCESS) return false;

    // Profiling information:
    cl_ulong timeStart = profileEvent.getProfilingInfo< CL_PROFILING_COMMAND_START >(&error);
    cl_ulong timeEnd = profileEvent.getProfilingInfo< CL_PROFILING_COMMAND_END   >(&error);
    if (error != CL_SUCCESS) return false;

    long double elapsed = static_cast<long double>(timeEnd - timeStart);
    long double elapsedMs = elapsed * 1e-6;
    out << ',' << threadsPerBlock << ',' << elapsedMs;

    float maxOpenCLError = 0.f;
    auto inIt = groundTruth.begin(), outIt = host_output.begin();
    for (; inIt != groundTruth.end(); ++inIt, ++outIt) {
      maxOpenCLError = std::fmaxf(maxOpenCLError, std::fabsf(*inIt - *outIt));
    }
    if (maxOpenCLError > 1e-5) {
      std::cerr << "Max error with OpenCL execution is: " << maxOpenCLError << '\n';
    }
    return true;
  };


  std::vector< cl::Platform > platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    std::cerr << "No OpenCL platforms could be detected!\n";
  } else {
    if(printSpecs) std::cout << "[INFO] Detected " << platforms.size() << " OpenCL platform(s).\n";
    for (auto const& platform : platforms) {
      if(printSpecs) std::cout << "[INFO]   '" << platform.getInfo< CL_PLATFORM_NAME >() << "'\n";

      std::vector< cl::Device > devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

      if (!devices.empty()) {
        if(printSpecs) std::cout << "[INFO]     Detected " << devices.size() << " device(s) on this platform:\n";
        for (auto const& device : devices) {
          if (printSpecs) {
            std::string constMemUnit;
            float constMem = formatMemory(device.getInfo< CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE >(), constMemUnit);
            std::string globalMemUnit;
            float globalMem = formatMemory(device.getInfo< CL_DEVICE_GLOBAL_MEM_SIZE >(), globalMemUnit);
            std::string localMemUnit;
            float localMem = formatMemory(device.getInfo< CL_DEVICE_LOCAL_MEM_SIZE >(), localMemUnit);

            bool supportsImages = device.getInfo< CL_DEVICE_IMAGE_SUPPORT >();

            std::cout << "[INFO]       - Device name:            '" << device.getInfo< CL_DEVICE_NAME >() << "'\n";
            std::cout << "[INFO]       - Device global   memory: " << std::setw(3) << std::setprecision(3) << globalMem << ' ' << globalMemUnit << '\n';
            std::cout << "[INFO]       - Device constant memory: " << std::setw(3) << std::setprecision(3) << constMem << ' ' << constMemUnit << '\n';
            std::cout << "[INFO]       - Device local    memory: " << std::setw(3) << std::setprecision(3) << localMem << ' ' << localMemUnit << '\n';
            std::cout << "[INFO]       - Device supports images? " << std::boolalpha << supportsImages;
            if (supportsImages) {
              std::cout << " (max. dimensions: "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_WIDTH  >() << " x "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_HEIGHT >() << " x "
                << device.getInfo< CL_DEVICE_IMAGE3D_MAX_DEPTH  >() << ")";
            }
            std::cout << "\n";
          }

          for(int threadsPerBlock = 0; threadsPerBlock <= maxThreadsPerBlock; ++threadsPerBlock) {
            if(!callOpenCLKernel(device, 1ull << threadsPerBlock)) {
              goto AFTER_OPENCL;
            }
          }

#ifdef HAS_CUDA
          callOpenCLKernel(device, maxPotentialCUDABlockSizeDev0);
#endif // HAS_CUDA

          AFTER_OPENCL:;
        }
      }
    }
  }

  out << '\n';

#endif
}


double getMaxMemoryInGB(double maxMemoryGiB) {
  constexpr size_t toGiB = 1ull << 30;
  size_t maxMemoryInBytes = static_cast< size_t >(maxMemoryGiB * toGiB);

#ifdef HAS_CUDA
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  if (error == cudaError_t::cudaSuccess) {
    for (size_t freeMemory : availableMemoryPerDevice) {
      maxMemoryInBytes = std::min< size_t >(maxMemoryInBytes, freeMemory);
    }
  }
#endif

#ifdef HAS_OPENCL
  std::vector< cl::Platform > platforms;
  cl::Platform::get(&platforms);

  if (!platforms.empty()) {
    for (auto const& platform : platforms) {
      std::vector< cl::Device > devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

      if (!devices.empty()) {
        for (auto const& device : devices) {
          maxMemoryInBytes = std::min< size_t >(maxMemoryInBytes, device.getInfo< CL_DEVICE_GLOBAL_MEM_SIZE >());
        }
      }
    }
  }
#endif

  double maxMemoryInGB = static_cast< double >(maxMemoryInBytes) / (1ull << 30);
  maxMemoryInGB /= 2.0;  // Need both input and output size.
  maxMemoryInGB *= 0.75; // Need some more memory for the offsets. Do not take all of it.
  return maxMemoryInGB;
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


  double maxMemoryInGB = getMaxMemoryInGB(maxMemoryGiB);
  std::cout << "[INFO]  Max. memory used for input data: " << maxMemoryInGB << " GiB.\n";

  std::ofstream log(benchmarkLogFile);

  bool firstRun = true;
  int percentageStep = 100 / numPercentages;
  for (int percentage = 100; percentage >= 1; percentage -= percentageStep) {
    for(int K = Kmin; K <= Kmax; K += 2) {
      benchmark(log, percentage / 100.0 * maxMemoryInGB, K, firstRun, maxThreadsBlock);
      log << std::string(100, '=') << '\n';
      firstRun = false;
    }
  }

  return EXIT_SUCCESS;
}