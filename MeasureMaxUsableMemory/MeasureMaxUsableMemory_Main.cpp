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

bool testTimeout(double memInGiB, int K, std::string& errorMsg) {
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

  // ======================================== IMPORTANT NOTES =============================================== 
  // Create the buffer for the input data, i.e., the voxel data read from file.
  // Assume the following configuration where this was tested:
  //    * Dell Precision Tower  (Intel i7-7700K @ 4.2GHz)
  //    * Intel HD Graphics 630 (integrated)
  //    * NVIDIA Quadro M4000   (discrete)
  //    * Numeric data (e.g., 5GB of floats for both input and output)
  // 
  // If we would go the standard way, i.e., creating the buffer from a given host 
  // pointer and writing it to the device, then the following happens:
  //    1. The CPU part: Executes as planned.
  //    2. The CUDA part (only NVIDIA GPU) -> as planned:
  //       2.1 Data is held in CPU RAM       (5GB in RAM)
  //       2.2 Allocation for data on device (5GB on GPU)
  //       2.3 Execution of kernel on device
  //       2.4 Read-out and storage in pre-allocated buffer
  //    3. The OpenCL part (all 3 devices, the CPU, the integrated GPU and the discrete GPU)
  //       3.1 Data is held in CPU RAM       (5GB in RAM)
  //       3.2 Execution on NVIDIA device:
  //             3.2.1  Execution as above.
  //       3.3 Execution on Intel GPU and CPU via OpenCL:
  //             3.3.1  The buffer allocation on the device allocates it
  //                    in the CPU RAM (additional 5GB), since the CPU itself
  //                    and the integrated card share their memory, i.e., the
  //                    device memory regions are NOT separated.
  // 
  //                    In total, there would be the double memory overhead (10GB instead of 5)
  //                    although the data is already in RAM but not "fast" accessible by the device
  //                    (-> pinned memory), and an additional copy operation which takes some time.
  // 
  // So, to accomodate also the integrated devices without requiring double the memory, we make use
  // of the mapped memory concept also available in OpenCL.
  // Specifically, we need to allocate the cl::Buffer instances that will hold the data. Each such cl::Buffer
  // has two parts: the CPU part and the GPU part. Now, in order to allow the driver of the integrated GPUs 
  // (in our case the Intel driver) to make use of the zero-copy principle, i.e., to make use of the 
  // data to be held in RAM, it is easiest[^1] to let OpenCL allocate the buffers for us.
  // Therefore, we pass the flag CL_MEM_ALLOC_HOST_PTR, which will allocate the requested memory on the device
  // having all the proper alignment. The allocation itself does not do anything yet.
  // Then, we make use of the memory mapping mechanism. Specifically, we take the (OpenCL-)allocated buffer
  // and map it into the host address space (marked for write). This yields a native pointer, into which we
  // read the data. For the data to be accessible on the device, we must first unmap that pointer.
  // After the kernel was executed, we aim to read out the data. For that, we also allocated such a cl::Buffer
  // and map the result data on the device into the host address space, which again yields a native pointer
  // which can be read from. After reading, we need to unmap it again.
  // Using this mapped-memory business, we only need the space in RAM we would need conventionally, i.e., the
  // input and output data, corresponding to 5GB in our example, also when executing on the integrated device.
  //
  // [^1] Another possible way (specific to Intel) would be to allocate the host memory part respecting a
  //      certain alignment (currently on Intel: 4096 byte). Then the driver might recognize this special alignment
  //      and would also go for the zero-copy version. However, the aligned allocation functions either require
  //      special allocation functions specific to operating systems (_aligned_alloc, posix_memalign, ...),
  //      modern concepts such as std::aligned_alloc (standard in C++17, but not on our in principle fully C++17 compatible
  //      hardware ...), or a manual implementation keeping track of the original pointers to free them accordingly.
  //      Still, the alignment would be compatible only/mostly to Intel.
  //      Letting OpenCL allocate the buffers hides this problem as OpenCL gives hardware-specific implementations there.
  //      More details: https://www.intel.com/content/www/us/en/developer/articles/training/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics.html 


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
      // Launch the CUDA kernel, see the "launchKernel" function in "CommonKernels.cu" for details.
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

#define HANDLE_OCL_ERROR(line) if (error != CL_SUCCESS) { errorMsg = getOpenCLError(error, line); return false; }

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
      std::stringstream errMsg;
      errMsg << "[CL] Error while building OpenCL code: " << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >(device);
      errorMsg = errMsg.str();
      return false;
    }
    cl_int error = CL_SUCCESS;

    // See the above large comment for a description of the following mapped-memory code.

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

    error |= queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
    HANDLE_OCL_ERROR(__LINE__);

    error |= queue.finish();
    HANDLE_OCL_ERROR(__LINE__);

    {
      float* host_output_mapped = static_cast< float* >(queue.enqueueMapBuffer(deviceOut, CL_TRUE, CL_MAP_READ, 0, sizeInBytes, nullptr, nullptr, &error));
      HANDLE_OCL_ERROR(__LINE__);

      if(host_output_mapped[0] > 1e-5) {
        // According to our kernel, the padding area (to which this very first element belongs) must be zero.
        errorMsg = "[CL] The very first value should have been zero!";
        return false;
      }

      error |= queue.enqueueUnmapMemObject(deviceOut, host_output_mapped);
      HANDLE_OCL_ERROR(__LINE__);
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

  maxMemory = getMaxMemoryInGiB(std::max(minMemory, maxMemory));
  double actualMinMemoryGiB = std::min(minMemory, maxMemory);
  double actualMaxMemoryGiB = std::max(minMemory, maxMemory);
  double memoryStep         = (actualMaxMemoryGiB - actualMinMemoryGiB) / numSteps;
  double currentMemoryGiB   = actualMinMemoryGiB;

  std::cout << "Trying the range from " << actualMinMemoryGiB << " to " << actualMaxMemoryGiB << " GiB.\n\n";

  std::string errorMsg;

  for(int step = 0; step <= numSteps; ++step) {
    if(step > 0) {
      currentMemoryGiB += memoryStep;

      // Sleep for some time to let the GPUs relax a little bit (temperature gets high!).
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    std::string unit;
    double currentMemPrint = formatMemory(currentMemoryGiB * (1ull << 30), unit);
    std::cout << "  Testing with " << std::setw(4) << std::fixed << std::setprecision(4) << currentMemPrint << ' ' << unit << " ... ";

    if(!testTimeout(currentMemoryGiB, maxK, errorMsg)) {
      std::cout << "done.\n";
      std::cerr << "  -> Error occured: " << errorMsg << '\n';
      break;
    }

    std::cout << "done.\n";
  }

  return EXIT_SUCCESS;
}