
#ifdef HAS_CUDA

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
  int localIndex  = threadIdx.x;

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


int launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTime, int threadsPerBlock)
{
  cudaError_t error;
  size_t sizeInBytes = N * sizeof(float);
  size_t offsetBytes = K * K * K * sizeof(size_t);
  
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
  error = cudaMalloc((void**)&device_offsets, offsetBytes);
  if(error != cudaError_t::cudaSuccess) {
    cudaFree(device_in);
    cudaFree(device_out);
    return -1;
  }

  cudaMemcpy(device_in,           in, sizeInBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_offsets, offsets, offsetBytes, cudaMemcpyHostToDevice);

  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  useStats<<< blocksPerGrid, threadsPerBlock >>>(device_in, device_out, device_offsets, K, N, dimX, dimY, dimZ);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(elapsedTime, start, stop);

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
