
#ifdef HAS_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MaskBasedProcessing_OpenCL_vs_CUDA.h"

__global__
void avg(float* in, float* out, size_t* offsets, int K, size_t N, size_t dimX, size_t dimY, size_t dimZ) {
  int K2 = K >> 1;

  for (size_t i = 0; i < N; ++i) {
    size_t z = i / (dimY * dimX);
    size_t j = i - z * dimY * dimX;
    size_t y = j / dimX;
    size_t x = j % dimX;

    if (x < K2 || y < K2 || z < K2 || x > dimX - 1 - K2 || y > dimY - 1 - K2 || z > dimZ - 1 - K2) {
      out[i] = 0.f;
    } else {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += in[i + offsets[k]];
      }
      out[i] = sum / (K * K * K);
    }
  }
}


int launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ)
{
  cudaError_t error;
  size_t sizeInBytes = N * sizeof(float);
  
  float* device_in;
  error = cudaMalloc((void**)&device_in, sizeInBytes);
  if(error != cudaError_t::cudaSuccess) {
    return -1;
  }
  
  float* device_out;
  error = cudaMalloc((void**)&device_out, sizeInBytes);
  if(error != cudaError_t::cudaSuccess) {
    cudaFree(device_in);
    return -1;
  }
  
  size_t* device_offsets;
  error = cudaMalloc((void**)&device_offsets, K * sizeof(size_t));
  if(error != cudaError_t::cudaSuccess) {
    cudaFree(device_in);
    cudaFree(device_out);
    return -1;
  }

  cudaMemcpy(device_in,           in,        sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_offsets, offsets, K * sizeof(size_t), cudaMemcpyHostToDevice);

  avg<<< 1, 1 >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  cudaDeviceSynchronize();

  cudaMemcpy(out, device_out, sizeInBytes, cudaMemcpyDeviceToHost);

  cudaFree(device_in);
  cudaFree(device_out);
  cudaFree(device_offsets);
  return 0;
}


void getGPUInformation(int* nr_gpus, size_t** availableMemoryPerDevice, size_t** totalMemoryPerDevice) {
  cudaGetDeviceCount(nr_gpus);
  *availableMemoryPerDevice = (size_t*)malloc(*nr_gpus * sizeof(size_t));
  *totalMemoryPerDevice     = (size_t*)malloc(*nr_gpus * sizeof(size_t));

  for (int gpuID = 0; gpuID < *nr_gpus; ++gpuID) {
    cudaSetDevice(gpuID);
    int _id = 0;
    cudaGetDevice(&_id);
    cudaMemGetInfo(availableMemoryPerDevice[gpuID], totalMemoryPerDevice[gpuID]);
  }
}

#endif // HAS_CUDA
