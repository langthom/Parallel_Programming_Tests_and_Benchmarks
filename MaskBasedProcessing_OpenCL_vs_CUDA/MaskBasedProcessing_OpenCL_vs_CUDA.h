#ifndef MBPT_H
#define MBPT_H

#ifdef HAS_CUDA

#include <cuda_runtime.h> // for definition of __global__

__global__ void avg(float* in, float* out, size_t* offsets, int K);

int launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ);

void getGPUInformation(int* nr_gpus, size_t** availableMemoryPerDevice, size_t** totalMemoryPerDevice);

#endif

#endif