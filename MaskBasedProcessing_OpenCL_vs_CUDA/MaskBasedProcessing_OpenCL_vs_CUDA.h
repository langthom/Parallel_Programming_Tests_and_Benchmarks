#ifndef MBPT_H
#define MBPT_H

#ifdef HAS_CUDA

#include <cuda_runtime.h> // for definition of __global__

cudaError_t launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTimeInMilliseconds, int threadsPerBlock);

cudaError_t getGPUInformation(int& nr_gpus, std::vector< size_t >& availableMemoryPerDevice, std::vector< size_t >& totalMemoryPerDevice);

#endif

#endif