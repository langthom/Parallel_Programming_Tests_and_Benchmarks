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
#ifndef Parallel_Programming_Tests__CUDA_MULTI_GPU__H
#define Parallel_Programming_Tests__CUDA_MULTI_GPU__H

#ifdef HAS_CUDA
#include <cuda_runtime.h> // for definition of cudaError_t

/// <summary>
/// Computes the maximum allocation size given a multi-CUDA-GPU system.
/// That is, it computes the total available device memory, and returns the percentage to
/// actually use of the minimum of the maximum memory limit and the avaible device memory.
/// </summary>
/// <param name="maxMemoryInGiB">Maximum memory to use for invocation, in GiB.</param>
/// <param name="actuallyUsePercentage">Percentage of memory to actually use. Defaults to 0.9, i.e., 90%.</param>
/// <returns>The maximum allocation size for the benchmark(s) in GiB.</returns>
double getMaxAllocationSizeMultiCUDAGPU(double maxMemoryInGiB, double actuallyUsePercentage = 0.9);

/// <summary>
/// Launches the statistics kernel in a multi-CUDA-GPU setting.
/// Specifically, it takes the given chunk of memory and further splits it up according to the avaible
/// CUDA devices with a focus on keeping the load on all devices approximately equal (in percents). Then,
/// the chunks are copied to the devices, processed, and written back to the output buffer without any padding.
/// </summary>
/// <param name="out">The output buffer, must be properly allocated to allow writing all data within the padding to it.</param>
/// <param name="in">The input buffer to read from.</param>
/// <param name="N">The number of voxels within the input volume.</param>
/// <param name="offsets">The calculated offsets for jumping in the input memory blob.</param>
/// <param name="K">The environment side length.</param>
/// <param name="dimX">The tertiary dimension of the input volume.</param>
/// <param name="dimY">The secondary dimension of the input volume.</param>
/// <param name="dimZ">The primary dimension of the input volume.</param>
/// <param name="elapsedTime">The elapsed GPU time, i.e., the maximum time each GPU spent computing.</param>
/// <param name="threadsPerBlock">The threads per block to use.</param>
/// <returns>Returns the CUDA error which occured, if any, or cudaError_t::cudaSuccess on success.</returns>
cudaError_t launchKernelMultiCUDAGPU(float* out,
                                     float* in,
                                     int64_t N,
                                     int64_t* offsets,
                                     int K,
                                     int64_t dimX,
                                     int64_t dimY,
                                     int64_t dimZ,
                                     float* elapsedTime,
                                     int threadsPerBlock);

#endif // HAS_CUDA

#endif // Parallel_Programming_Tests__CUDA_MULTI_GPU__H