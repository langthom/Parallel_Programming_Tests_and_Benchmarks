
#ifdef HAS_CUDA

#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaskBasedProcessing_OpenCL_vs_CUDA.h"


__device__
void stats(float* features, float* in, int globalIndex, size_t* offsets, int envSize) {
  float count = 0, mean = 0, M2 = 0, M3 = 0, M4 = 0;
  for (int voxelIndex = 0; voxelIndex < envSize; ++voxelIndex) {
    float voxel = in[globalIndex + offsets[voxelIndex]];
    float n1 = count;
    count += 1;
    float delta = voxel - mean;
    float delta_n = delta / count;
    float delta_n2 = delta_n * delta_n;
    float term1 = delta * delta_n * n1;
    mean += delta_n;
    M4 += term1 * delta_n2 * (count * count - 3 * count + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
    M3 += term1 * delta_n * (count - 2) - 3 * delta_n * M2;
    M2 += term1;
  }

  *features++ = mean;
  *features++ = powf(M2 / count, 0.5f);
  *features++ = M2 < 1e-7 ? 0.0 : (M3 * powf(count, 0.5f) / powf(M2, 1.5f));
  *features   = M2 < 1e-7 ? -3.0 : ((M4 * count) / (M2 * M2) - 3);
}


__global__
void useStats(float* in, float* out, size_t* offsets, int const K, size_t N, size_t dimX, size_t dimY, size_t dimZ) {
  int K2 = K >> 1;
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if(globalIndex < N) {
    int z = globalIndex / (dimY * dimX);
    int j = globalIndex - z * dimY * dimX;
    int y = j / dimX;
    int x = j % dimX;

    if (x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2) {
      out[globalIndex] = 0.f;
    } else {
      float features[4];
      stats(features, in, globalIndex, offsets, K*K*K);
      out[globalIndex] = features[0]; //!!
    }
  }
}


cudaError_t launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTime, int threadsPerBlock)
{
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  cudaError_t error;
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


cudaError_t getGPUInformation(int& nr_gpus, std::vector< size_t >& availableMemoryPerDevice, std::vector< size_t >& totalMemoryPerDevice) {
  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaGetDeviceCount(std::addressof(nr_gpus));
  if(error != cudaError_t::cudaSuccess || nr_gpus <= 0) {
    // Either on error or if there are no usable GPUS, early return.
    return error;
  }

  availableMemoryPerDevice.resize(nr_gpus);
  totalMemoryPerDevice.resize(nr_gpus);

  // If there are GPUs available for CUDA usage, retrieve the info from them.
  for (int gpuID = 0; gpuID < nr_gpus; ++gpuID) {
    error = cudaSetDevice(gpuID);
    error = cudaMemGetInfo(&availableMemoryPerDevice[gpuID], &totalMemoryPerDevice[gpuID]);

    if(error != cudaError_t::cudaSuccess) {
      return error;
    }
  }

  return cudaError_t::cudaSuccess;
}

#endif // HAS_CUDA
