
#define NOMINMAX

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>

#ifdef HAS_CUDA
#include "../CommonKernels/CommonKernels.h"
#endif

#ifdef HAS_OPENCL
#include <CL/cl.hpp>
#endif

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

void readData(float* buf, size_t N) {
  std::fill_n(buf, N, 1.f);
}

bool testTimeout(double memInGiB, int K, std::string& errorMsg) {
  int K2 = K >> 1;
  int E = 9;

  size_t dimX = 1ull << E;
  size_t dimY = 1ull << E;
  size_t dimZ = (size_t)(memInGiB * (1ull << 30)) / (2/* in and output */ * sizeof(float) * dimY * dimX);
  size_t N = dimZ * dimY * dimX;
  size_t sizeInBytes = N * sizeof(float);

  size_t envSize = static_cast< size_t >(K) * K * K;
  std::vector< size_t > offsets(envSize, 0);
  size_t _o = 0;
  for (int _z = -K2; _z <= K2; ++_z) {
    for (int _y = -K2; _y <= K2; ++_y) {
      for (int _x = -K2; _x <= K2; ++_x) {
        offsets[_o++] = (_z * dimY + _y) * dimX + _x;
      }
    }
  }

  // Call the CUDA kernel
#ifdef HAS_CUDA
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  if (error != cudaError_t::cudaSuccess) {
    std::stringstream errMsg;
    errMsg << "[CUDA] Error during GPU information retrieval, error was: " << cudaGetErrorString(error);
    errorMsg = errMsg.str();
    return false;
  } else {
    std::vector< float > data(N, 0);
    std::vector< float > host_output(N, 0);
    readData(data.data(), N);

    auto call_cuda_kernel = [&](int deviceID, int threadsPerBlock) -> bool {
      float elapsedTimeInMilliseconds = -1;
      cudaError_t cuda_kernel_error = launchKernel(host_output.data(), data.data(), N, offsets.data(), K, dimX, dimY, dimZ, &elapsedTimeInMilliseconds, deviceID, threadsPerBlock);
      if (cuda_kernel_error != cudaError_t::cudaSuccess) {
        std::stringstream errMsg;
        errMsg << "[CUDA] Error during Kernel launch, error was '" << cudaGetErrorString(cuda_kernel_error) << "'";
        errorMsg = errMsg.str();
        return false;
      }
      return true;
    };

    for(int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
      if(!call_cuda_kernel(gpuID, 64)) {
        return false;
      }
    }
  }
#endif

  // Call the OpenCL kernel
#ifdef HAS_OPENCL

#define HANDLE_OCL_ERROR if (error != CL_SUCCESS) { errorMsg = getOpenCLError(error); return false; }

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
      std::stringstream errMsg;
      errMsg << "Error while building OpenCL code: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device);
      errorMsg = errMsg.str();
      return false;
    }
    cl_int error = CL_SUCCESS;

    cl::Buffer deviceIn( context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeInBytes, nullptr, &error);
    HANDLE_OCL_ERROR;
    cl::Buffer deviceOut(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeInBytes, nullptr, &error);
    HANDLE_OCL_ERROR;
    cl::Buffer deviceOff(context, CL_MEM_READ_ONLY,  envSize * sizeof(size_t), nullptr, &error);
    HANDLE_OCL_ERROR;

    {
      float* input_mapped = static_cast< float* >(queue.enqueueMapBuffer(deviceIn, CL_TRUE, CL_MAP_WRITE, 0, sizeInBytes, nullptr, nullptr, &error));
      HANDLE_OCL_ERROR;
      readData(input_mapped, N);
      error |= queue.enqueueUnmapMemObject(deviceIn, input_mapped);
      HANDLE_OCL_ERROR;
    }

    error |= queue.enqueueWriteBuffer(deviceOff, CL_FALSE, 0, envSize * sizeof(size_t), offsets.data());
    HANDLE_OCL_ERROR;

    cl::Kernel kernel(program, "useStats");
    error |= kernel.setArg(0, deviceIn);
    error |= kernel.setArg(1, deviceOut);
    error |= kernel.setArg(2, deviceOff);
    error |= kernel.setArg(3, K);
    error |= kernel.setArg(4, N);
    error |= kernel.setArg(5, dimX);
    error |= kernel.setArg(6, dimY);
    error |= kernel.setArg(7, dimZ);
    HANDLE_OCL_ERROR;

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
    HANDLE_OCL_ERROR;

    {
      float* host_output_mapped = static_cast< float* >(queue.enqueueMapBuffer(deviceOut, CL_TRUE, CL_MAP_READ, 0, sizeInBytes, nullptr, nullptr, &error));
      HANDLE_OCL_ERROR;

      if(host_output_mapped[0] > 1e-5) {
        // According to our kernel, the padding area (to which this very first element belongs) must be zero.
        errorMsg = "[CL] The very first value should have been zero!";
        return false;
      }

      error |= queue.enqueueUnmapMemObject(deviceOut, host_output_mapped);
      HANDLE_OCL_ERROR;
    }
    return true;
  };

#undef HANDLE_OCL_ERROR

  std::vector< cl::Platform > platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) {
    errorMsg = "[CL] No OpenCL platforms could be detected!";
    return false;
  } else {
    for (auto const& platform : platforms) {
      std::vector< cl::Device > devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

      if (!devices.empty()) {
        for (auto const& device : devices) {
          if(!callOpenCLKernel(device, 64)) {
            return false;
          }
        }
      }
    }
  }
#endif

  return true;
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
  maxMemoryInGB *= 0.90; // Need some more memory for the offsets. Do not take all of it.
  return maxMemoryInGB;
}

/* ------------------------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char** argv) {

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <min-memory-GiB>  <max-memory-GiB>  <num-memory-steps>  <max-K>\n\n";
    std::cerr << "  <min-memory-GiB>                Minimum memory to test (in GiB, float).\n";
    std::cerr << "  <max-memory-GiB>                Maximum memory to test (in GiB, float). If the device does not have sufficient memory, this will be reduced automatically.\n";
    std::cerr << "  <num-memory-steps>              Number of memory steps in [<min-memory-GiB>, <max-memory-GiB>] to test.\n";
    std::cerr << "  <max-K>                         Maximum environment size to test, must be bigger than one and odd.\n";
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

  double minMemory = parseDouble(argv[1]);
  double maxMemory = parseDouble(argv[2]);
  int numSteps     = parseInt(argv[3]);
  int maxK         = parseInt(argv[4]);

  if(maxK % 2 == 0 || maxK < 3) {
    std::cerr << "Error: Environment sizes need to be bigger than one and odd!\n";
    return EXIT_FAILURE;
  }

  maxMemory = getMaxMemoryInGB(std::max(minMemory, maxMemory));
  double actualMinMemoryGiB = std::min(minMemory, maxMemory);
  double actualMaxMemoryGiB = std::max(minMemory, maxMemory);
  double memoryStep         = (actualMaxMemoryGiB - actualMinMemoryGiB) / numSteps;
  double currentMemoryGiB   = actualMinMemoryGiB;

  std::cout << "Trying the range from " << actualMinMemoryGiB << " to " << actualMaxMemoryGiB << " GiB.\n\n";

  std::string errorMsg;

  for(int step = 0; step <= numSteps; ++step) {
    if(step > 0) {
      currentMemoryGiB += memoryStep;
    }

    std::string unit;
    double currentMemPrint = formatMemory(currentMemoryGiB * (1ull << 30), unit);
    std::cout << "  Testing with " << std::setw(4) << std::fixed << std::setprecision(4) << currentMemPrint << ' ' << unit << " ... ";

    if(!testTimeout(currentMemoryGiB, maxK, errorMsg)) {
      std::cout << "done.\n";
      std::cerr << "  -> Error occured: " << errorMsg << '\n';
      break;
    }

    // Sleep for 500ms to let the GPUs relax a little bit (temperature gets high!).
    std::this_thread::sleep_for(std::chrono::milliseconds(700));

    std::cout << "done.\n";
  }

  std::string unit;
  double finalMem = formatMemory(currentMemoryGiB * (1ull << 30), unit);
  std::cout << "[FINAL] Max usable memory before timeout is " << finalMem << ' ' << unit << '\n';

  return EXIT_SUCCESS;
}