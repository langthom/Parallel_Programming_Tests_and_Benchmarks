#ifndef MBPT_H
#define MBPT_H

#ifdef HAS_CUDA

#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>      // for definition of cl_int

#include <cuda_runtime.h> // for definition of __global__

cudaError_t launchKernel(float* out, float* in, size_t N, size_t* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTimeInMilliseconds, int deviceID, int threadsPerBlock);

cudaError_t getGPUInformation(int& nr_gpus, std::vector< std::string >& deviceNames, std::vector< size_t >& availableMemoryPerDevice, std::vector< size_t >& totalMemoryPerDevice);

cudaError_t getMaxPotentialBlockSize(int& maxPotentialBlockSize, int deviceID);

std::string getOpenCLKernel();

std::string getOpenCLError(cl_int errorCode);

#endif // HAS_CUDA

#endif // MBPT_H