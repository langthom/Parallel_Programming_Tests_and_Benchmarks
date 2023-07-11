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
#ifndef Parallel_Programming_Tests__CommonKernels__H
#define Parallel_Programming_Tests__CommonKernels__H

#include <vector>

#ifdef HAS_CUDA
#include <cuda_runtime.h> // for definition of __global__
#endif // HAS_CUDA

#ifdef HAS_OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.hpp>      // for definition of cl_int
#endif // HAS_OPENCL

/* ================================================ General stuff ================================================ */

double getMaxMemoryInGiB(double maxMemoryGiB, double actuallyUsePercentage = 0.90);

/* ================================================  CUDA  stuff ================================================= */
#ifdef HAS_CUDA

cudaError_t launchKernel(float* out, float* in, size_t N, int* offsets, int K, size_t dimX, size_t dimY, size_t dimZ, float* elapsedTimeInMilliseconds, int deviceID, int threadsPerBlock);

cudaError_t getGPUInformation(int& nr_gpus, std::vector< std::string >& deviceNames, std::vector< size_t >& availableMemoryPerDevice, std::vector< size_t >& totalMemoryPerDevice);

cudaError_t getMaxPotentialBlockSize(int& maxPotentialBlockSize, int deviceID);

#endif // HAS_CUDA

/* ================================================ OpenCL stuff ================================================= */
#ifdef HAS_OPENCL

std::string getOpenCLKernel();

std::string getOpenCLError(cl_int errorCode, int line);

#endif // HAS_OPENCL

#endif // Parallel_Programming_Tests__CommonKernels__H