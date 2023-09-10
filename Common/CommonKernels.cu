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

#include "CommonKernels.h"

#include <cassert>
#include <map>
#include <vector>
#include <string>

#ifdef HAS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "StatisticsKernel.h"
#endif // HAS_CUDA

#ifdef HAS_OPENCL
#include <CL/cl.hpp>
#endif // HAS_OPENCL


/* ================================================ General stuff ================================================ */

double getMaxMemoryInGiB(double maxMemoryGiB, double actuallyUsePercentage, bool respectOpenCLLimit) {
  constexpr size_t toGiB = 1ull << 30;
  size_t maxMemoryInBytes = static_cast< size_t >(maxMemoryGiB * toGiB);

#ifdef HAS_CUDA
  int nr_gpus = 0;
  std::vector< std::string > deviceNames;
  std::vector< size_t > availableMemoryPerDevice, totalMemoryPerDevice;
  cudaError_t error = getGPUInformation(nr_gpus, deviceNames, availableMemoryPerDevice, totalMemoryPerDevice);

  // Note that there doesn't seem to exist a rule regarding the maximum allocation on 
  // CUDA devices, but rather many sources recommend small sizes or an iterative approach
  // in order to check the limit per device.
  // But, according to some CUDA release notes[^1], the maximum size of a single allocation
  // created by cudaMalloc or similar functions is limited to
  // 
  //    max( (system memory size in MB  -  512 MB )/2,  PAGING_BUFFER_SEGMENT_SIZE )
  // 
  // where PAGING_BUFFER_SEGMENT_SIZE is specified on Windows Vista to be roughly 2 GB.
  // 
  // Yet, it seems that there are really no official limits to memory allocation, and that the 
  // OpenCL limit queried below typically lies around the range of 25% of the available memory
  // for many common devices[^2].
  // 
  // [^1] https://forums.developer.nvidia.com/t/how-much-gpu-memory-can-cudamalloc-get/19796/7
  // [^2] https://forums.developer.nvidia.com/t/why-is-cl-device-max-mem-alloc-size-never-larger-than-25-of-cl-device-global-mem-size-only-on-nvidia/47745/6 

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
          // Note that OpenCL does not allow for the allocation of large chunks of memory, but only
          // for a vendor-specific limit identified per device by CL_DEVICE_MAX_MEM_ALLOC_SIZE.
          // Please also note that this size is *considerably* smaller than the available memory:
          // On the tested device(s), the maximum available memory was about 7 GiB, yet the allocatable
          // memory only amounted to roughly 1.3 GiB.
          cl_ulong maxMemoryOCL = respectOpenCLLimit ? device.getInfo< CL_DEVICE_MAX_MEM_ALLOC_SIZE >() : device.getInfo< CL_DEVICE_GLOBAL_MEM_SIZE >();
          maxMemoryInBytes = std::min< size_t >(maxMemoryInBytes, maxMemoryOCL);
        }
      }
    }
  }
#endif

  double maxMemoryInGB = static_cast< double >(maxMemoryInBytes) / (1ull << 30);
  maxMemoryInGB *= actuallyUsePercentage;
  return maxMemoryInGB;
}

/* ================================================  CUDA  stuff ================================================= */

#ifdef HAS_CUDA

cudaError_t launchKernel(float* out, float* in, int64_t N, int64_t* offsets, int K, int64_t dimX, int64_t dimY, int64_t dimZ, float* elapsedTime, int deviceID, int threadsPerBlock)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  assert(out);
  assert(in);
  assert(offsets);
  assert(elapsedTime);

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  HANDLE_ERROR(error);

  int pad = K - 1;
  int64_t sizeInBytes  = N * sizeof(float);
  int64_t sizeOutBytes = (dimX - pad) * (dimY - pad) * (dimZ - pad) * sizeof(float);
  int64_t offsetBytes  = K * K * K * sizeof(int64_t);
  
  float* device_in;
  error = cudaMalloc((void**)&device_in, sizeInBytes);
  HANDLE_ERROR(error);
  
  float* device_out;
  error = cudaMalloc((void**)&device_out, sizeInBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in));
  
  int64_t* device_offsets;
  error = cudaMalloc((void**)&device_offsets, offsetBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in); cudaFree(device_out));

  // Create events for timing the code. This shall include also the memory transfer to be 
  // comparable with the multi GPU and CPU approaches.
  cudaEvent_t start, stop;
  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);
  HANDLE_ERROR(error);

  error = cudaMemcpy(device_in,           in, sizeInBytes, cudaMemcpyHostToDevice);
  error = cudaMemcpy(device_offsets, offsets, offsetBytes, cudaMemcpyHostToDevice);
  HANDLE_ERROR(error);

  int64_t blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  statisticsKernel<<< blocksPerGrid, threadsPerBlock >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  error = cudaMemcpy(out, device_out, sizeOutBytes, cudaMemcpyDeviceToHost);
  HANDLE_ERROR(error);

  error = cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);
  error = cudaEventElapsedTime(elapsedTime, start, stop);
  HANDLE_ERROR(error);

  error = cudaFree(device_in);
  error = cudaFree(device_out);
  error = cudaFree(device_offsets);
  HANDLE_ERROR(error);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}


cudaError_t getGPUInformation(int& nr_gpus, std::vector< std::string >& deviceNames, std::vector< size_t >& availableMemoryPerDevice, std::vector< size_t >& totalMemoryPerDevice) {
  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaGetDeviceCount(std::addressof(nr_gpus));
  if(error != cudaError_t::cudaSuccess || nr_gpus <= 0) {
    // Either on error or if there are no usable GPUS, early return.
    return error;
  }

  deviceNames.resize(nr_gpus);
  availableMemoryPerDevice.resize(nr_gpus);
  totalMemoryPerDevice.resize(nr_gpus);

  // If there are GPUs available for CUDA usage, retrieve the info from them.
  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    error = cudaSetDevice(gpuID);
    error = cudaMemGetInfo(&availableMemoryPerDevice[gpuID], &totalMemoryPerDevice[gpuID]);
    
    cudaDeviceProp deviceProperties;
    error = cudaGetDeviceProperties(&deviceProperties, gpuID);
    deviceNames[gpuID] = deviceProperties.name;

    if(error != cudaError_t::cudaSuccess) {
      return error;
    }
  }

  return cudaError_t::cudaSuccess;
}

cudaError_t getMaxPotentialBlockSize(int& maxPotentialBlockSize, int deviceID) {
  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  if(error != cudaError_t::cudaSuccess) {
    return error;
  }

  int minBlocksPerGrid;
  error = cudaOccupancyMaxPotentialBlockSize(&minBlocksPerGrid, std::addressof(maxPotentialBlockSize), statisticsKernel);
  return error;
}

#endif // HAS_CUDA

/* ================================================ OpenCL stuff ================================================= */
#ifdef HAS_OPENCL

std::string getOpenCLKernel() {
  static std::string kernel = 
    "typedef long int64_t;\n\n"
    "void stats(float* features, __global float* in, int64_t globalIndex, __global int64_t* offsets, int envSize) {\n"
    "  float sum = 0.f;\n\n"
    "  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {\n"
    "    sum += in[globalIndex + offsets[voxelIndex]];\n"
    "  }\n"
    "  float mean = sum / envSize;\n\n"
    "  sum = 0.f;\n"
    "  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {\n"
    "    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;\n"
    "    sum += voxelDiff * voxelDiff;\n"
    "  }\n"
    "  float stdev = sqrt(sum / (envSize - 1));\n\n"
    "  sum = 0.f;\n"
    "  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {\n"
    "    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;\n"
    "    sum += voxelDiff * voxelDiff * voxelDiff;\n"
    "  }\n"
    "  float skewness = sum / (envSize * stdev * stdev * stdev);\n\n"
    "  sum = 0.f;\n"
    "  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {\n"
    "    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;\n"
    "    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;\n"
    "  }\n"
    "  float kurtosis = sum / (envSize * envSize * envSize) - 3.f;\n\n"
    "  features[0] = mean;\n"
    "  features[1] = stdev;\n"
    "  features[2] = skewness;\n"
    "  features[3] = kurtosis;\n"
    "}\n\n"
    "__kernel void useStats(__global float* in, __global float* out, __global int64_t* offsets, int K, int64_t N, int64_t dimX, int64_t dimY, int64_t dimZ) {\n"
    "  int K2 = K >> 1;\n"
    "  int64_t globalIndex = get_global_id(0);\n"
    "  int64_t dimY_withoutPadding = dimY - K + 1;\n"
    "  int64_t dimX_withoutPadding = dimX - K + 1;\n"
    "  float features[4];\n"
    "  int64_t z = globalIndex / (dimY * dimX);\n"
    "  int64_t j = globalIndex - z * dimY * dimX;\n"
    "  int64_t y = j / dimX;\n"
    "  int64_t x = j % dimX;\n"
    "  bool isInPadding = x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2;\n"
    "  if (!isInPadding) {"
    "    stats(features, in, globalIndex, offsets, K*K*K);\n"
    "    int64_t globalIndexWithoutPadding  = ((z - K2) * dimY_withoutPadding + (y - K2)) * dimX_withoutPadding + (x - K2);\n"
    "    out[globalIndexWithoutPadding] = features[0];\n"
    "  }\n"
    "}";

  return kernel;
}

std::string getOpenCLError(cl_int errorCode, int line) {
  static std::map< cl_int, const char* > codes{
    {CL_SUCCESS                                  , "No error"},
    {CL_DEVICE_NOT_FOUND                         , "Device not found"},
    {CL_DEVICE_NOT_AVAILABLE                     , "Device not available"},
    {CL_COMPILER_NOT_AVAILABLE                   , "Compiler not available"},
    {CL_MEM_OBJECT_ALLOCATION_FAILURE            , "Memory object allocation failed"},
    {CL_OUT_OF_RESOURCES                         , "Out of device resources"},
    {CL_OUT_OF_HOST_MEMORY                       , "Out of host memory"},
    {CL_PROFILING_INFO_NOT_AVAILABLE             , "Profiling information not available"},
    {CL_MEM_COPY_OVERLAP                         , "Memory copy: Overlap occured"},
    {CL_IMAGE_FORMAT_MISMATCH                    , "Image format mismatch"},
    {CL_IMAGE_FORMAT_NOT_SUPPORTED               , "Image format not supported"},
    {CL_BUILD_PROGRAM_FAILURE                    , "Building of program failed"},
    {CL_MAP_FAILURE                              , "Mapping failed"},
    {CL_MISALIGNED_SUB_BUFFER_OFFSET             , "Sub-buffer offset is mis-aligned"},
    {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "Status error for events in wait list"},
    {CL_COMPILE_PROGRAM_FAILURE                  , "Compiling the program failed"},
    {CL_LINKER_NOT_AVAILABLE                     , "Linker not available"},
    {CL_LINK_PROGRAM_FAILURE                     , "Linking the program failed"},
    {CL_DEVICE_PARTITION_FAILED                  , "Device partition failed"},
    {CL_KERNEL_ARG_INFO_NOT_AVAILABLE            , "Kernel argument info is not available"},
    {CL_INVALID_VALUE                            , "Invalid value"},
    {CL_INVALID_DEVICE_TYPE                      , "Invalid device type"},
    {CL_INVALID_PLATFORM                         , "Invalid platform"},
    {CL_INVALID_DEVICE                           , "Invalid device"},
    {CL_INVALID_CONTEXT                          , "Invalid context"},
    {CL_INVALID_QUEUE_PROPERTIES                 , "Invalid queue properties"},
    {CL_INVALID_COMMAND_QUEUE                    , "Invalid command queue"},
    {CL_INVALID_HOST_PTR                         , "Invalid host pointer"},
    {CL_INVALID_MEM_OBJECT                       , "Invalid memory object"},
    {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          , "Invalid image format descriptor"},
    {CL_INVALID_IMAGE_SIZE                       , "Invalid image size"},
    {CL_INVALID_SAMPLER                          , "Invalid sampler"},
    {CL_INVALID_BINARY                           , "Invalid binary"},
    {CL_INVALID_BUILD_OPTIONS                    , "Invalid build options"},
    {CL_INVALID_PROGRAM                          , "Invalid program"},
    {CL_INVALID_PROGRAM_EXECUTABLE               , "Invalid program executable"},
    {CL_INVALID_KERNEL_NAME                      , "Invalid kernel name"},
    {CL_INVALID_KERNEL_DEFINITION                , "Invalid kernel definition"},
    {CL_INVALID_KERNEL                           , "Invalid kernel"},
    {CL_INVALID_ARG_INDEX                        , "Invalid kernel argument index"},
    {CL_INVALID_ARG_VALUE                        , "Invalid kernel argument value"},
    {CL_INVALID_ARG_SIZE                         , "Invalid kernel argument size"},
    {CL_INVALID_KERNEL_ARGS                      , "Invalid kernel arguments"},
    {CL_INVALID_WORK_DIMENSION                   , "Invalid work dimension"},
    {CL_INVALID_WORK_GROUP_SIZE                  , "Invalid work group size"},
    {CL_INVALID_WORK_ITEM_SIZE                   , "Invalid work item size"},
    {CL_INVALID_GLOBAL_OFFSET                    , "Invalid global offset"},
    {CL_INVALID_EVENT_WAIT_LIST                  , "Invalid event wait list"},
    {CL_INVALID_EVENT                            , "Invalid event"},
    {CL_INVALID_OPERATION                        , "Invalid operation"},
    {CL_INVALID_GL_OBJECT                        , "Invalid OpenGL object"},
    {CL_INVALID_BUFFER_SIZE                      , "Invalid buffer size"},
    {CL_INVALID_MIP_LEVEL                        , "Invalid MIP level"},
    {CL_INVALID_GLOBAL_WORK_SIZE                 , "Invalid global work size"},
    {CL_INVALID_PROPERTY                         , "Invalid property"},
    {CL_INVALID_IMAGE_DESCRIPTOR                 , "Invalid image descriptor"},
    {CL_INVALID_COMPILER_OPTIONS                 , "Invalid compiler options"},
    {CL_INVALID_LINKER_OPTIONS                   , "Invalid linker options"},
    {CL_INVALID_DEVICE_PARTITION_COUNT           , "Invalid device partition count"},
  };
  return std::string("[CL] Error at line ") + std::to_string(line) + ": " + codes.at(errorCode);
}

#endif // HAS_OPENCL
