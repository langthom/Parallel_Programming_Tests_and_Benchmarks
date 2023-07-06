
#ifdef HAS_CUDA

#include <memory>
#include <vector>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaskBasedProcessing_OpenCL_vs_CUDA.h"


__device__
void stats(float* features, float* in, int globalIndex, size_t* offsets, int envSize) {
  // This is not an efficient or even numerically stable way to compute the centralized 
  // statistical moments, but it does a few more computations and thus put some mild
  // computational load on the device.

  float sum = 0.f;

  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    sum += in[globalIndex + offsets[voxelIndex]];
  }
  float mean = sum / envSize;

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff;
  }
  float stdev = sqrtf(sum / (envSize - 1));

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff;
  }
  float skewness = sum / (envSize * stdev * stdev * stdev);

  sum = 0.f;
  for(int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxelDiff = in[globalIndex + offsets[voxelIndex]] - mean;
    sum += voxelDiff * voxelDiff * voxelDiff * voxelDiff;
  }
  float kurtosis = sum / (envSize * stdev * stdev) - 3.f;

  features[0] = mean;
  features[1] = stdev;
  features[2] = skewness;
  features[3] = kurtosis;
}


__global__
void useStats(float* in, float* out, size_t* offsets, int const K, size_t N, size_t dimX, size_t dimY, size_t dimZ) {
  int K2 = K >> 1;
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride      = blockDim.x * gridDim.x;

  float features[4];

  for(int i = globalIndex; i < N; i += stride) {
    int z = i / (dimY * dimX);
    int j = i - z * dimY * dimX;
    int y = j / dimX;
    int x = j % dimX;

    bool isInPadding = x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2;

    if (isInPadding) {
      out[i] = 0.f;
    } else {
      stats(features, in, i, offsets, K*K*K);
      out[i] = features[0];
    }
  }
}

cudaError_t launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTime, int deviceID, int threadsPerBlock)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(deviceID);
  HANDLE_ERROR(error);

  size_t sizeInBytes = N * sizeof(float);
  size_t offsetBytes = K * K * K * sizeof(size_t);
  
  float* device_in;
  error = cudaMalloc((void**)&device_in, sizeInBytes);
  HANDLE_ERROR(error);
  
  float* device_out;
  error = cudaMalloc((void**)&device_out, sizeInBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in));
  
  size_t* device_offsets;
  error = cudaMalloc((void**)&device_offsets, offsetBytes);
  HANDLE_ERROR_STMT(error, cudaFree(device_in); cudaFree(device_out));

  error = cudaMemcpy(device_in,           in, sizeInBytes, cudaMemcpyHostToDevice);
  error = cudaMemcpy(device_offsets, offsets, offsetBytes, cudaMemcpyHostToDevice);
  HANDLE_ERROR(error);

  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  error = cudaEventCreate(&start);
  error = cudaEventCreate(&stop);
  error = cudaEventRecord(start, 0);
  HANDLE_ERROR(error);

  useStats<<< blocksPerGrid, threadsPerBlock >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  error = cudaEventRecord(stop, 0);
  error = cudaEventSynchronize(stop);
  error = cudaEventElapsedTime(elapsedTime, start, stop);
  HANDLE_ERROR(error);

  error = cudaDeviceSynchronize();
  HANDLE_ERROR(error);

  error = cudaMemcpy(out, device_out, sizeInBytes, cudaMemcpyDeviceToHost);
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
  error = cudaOccupancyMaxPotentialBlockSize(&minBlocksPerGrid, std::addressof(maxPotentialBlockSize), useStats);
  return error;
}

#endif // HAS_CUDA
