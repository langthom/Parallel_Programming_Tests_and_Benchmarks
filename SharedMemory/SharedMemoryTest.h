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
#ifndef Parallel_Programming_Tests__SharedMemoryTest__H
#define Parallel_Programming_Tests__SharedMemoryTest__H

#ifdef HAS_CUDA
#include <cuda_runtime.h> // for definition of cudaError_t

/// <summary>
/// Launches a 3D statistics kernel using the specified launch configuration, i.e.,
/// the specified number of threads per block in all three dimensions. For a 1D launch,
/// one can pass <em>threads = dim3(nthreads,1,1)</em>, while other dimensions are 
/// self-explanatory.
/// The according blocks per grid dimension are inferred automatically.
/// </summary>
/// <param name="out">Output buffer (without padding).</param>
/// <param name="in">Input buffer (with padding).</param>
/// <param name="N">Number of voxels.</param>
/// <param name="offsets">Computed offsets for region accessing.</param>
/// <param name="K">Environment size.</param>
/// <param name="dimX">Volume X dimension.</param>
/// <param name="dimY">Volume Y dimension.</param>
/// <param name="dimZ">Volume Z dimension.</param>
/// <param name="threads">Thread launch configuration.</param>
/// <param name="elapsedTime">Output buffer where to store the elapsed time of the kernel invocation.</param>
/// <param name="deviceID">Which device to use, if any.</param>
/// <returns>Returns any CUDA error that occured, if any.</returns>
cudaError_t launchKernelNDWithShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  unsigned int dimX, unsigned int dimY, unsigned int dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID);

/// \copydoc launchKernelNDWithShared
cudaError_t launchKernelNDWithoutShared(
  float* out, float* in,
  int_least64_t* offsets, int K, 
  unsigned int dimX, unsigned int dimY, unsigned int dimZ, 
  dim3 threads,
  std::vector< float >& elapsedMillis,
  int deviceID);


#endif // HAS_CUDA

#endif // Parallel_Programming_Tests__SharedMemoryTest__H