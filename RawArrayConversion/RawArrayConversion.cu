/*
* MIT License
* 
* Copyright (c) 2025 Dr. Thomas Lang
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

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void arrayConversionKernel(float* out, unsigned short* in, int N) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIndex < N) {
    out[globalIndex] = static_cast<float>(in[globalIndex]);
  }
}


cudaError_t cudaRawArrayConversion(
  float* out,
  unsigned short* in,
  long long N
) {
#define HANDLE_ERROR(err)             if(error != cudaError_t::cudaSuccess) { return error; }
#define HANDLE_ERROR_STMT(err, stmts) if(error != cudaError_t::cudaSuccess) { stmts; return error; }

  // The following block is exactly the same as our 1D launch in the CommonKernels header/implementation.

  cudaError_t error = cudaError_t::cudaSuccess;
  error = cudaSetDevice(0);
  HANDLE_ERROR(error);

  unsigned short* device_in;
  error = cudaMalloc((void**)&device_in, N * sizeof(unsigned short));
  HANDLE_ERROR(error);

  float* device_out;
  error = cudaMalloc((void**)&device_out, N * sizeof(float));
  HANDLE_ERROR_STMT(error, cudaFree(device_in));

  error = cudaMemcpy(device_in, in, N * sizeof(unsigned short), cudaMemcpyHostToDevice);
  HANDLE_ERROR(error);

  cudaEvent_t start, stop;
  error = cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
  error = cudaEventCreateWithFlags(&stop,  cudaEventBlockingSync);
  error = cudaEventRecord(start);

  int Ni = static_cast< int >(N);
  int threadsPerBlock = 512;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  arrayConversionKernel<<< blocksPerGrid, threadsPerBlock >>>(device_out, device_in, Ni);

  // Same as 1D again.
  error = cudaEventRecord(stop);
  error = cudaEventSynchronize(stop);

  error = cudaMemcpy(out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

  error = cudaFree(device_in);
  error = cudaFree(device_out);
  HANDLE_ERROR(error);
  return cudaError_t::cudaSuccess;

#undef HANDLE_ERROR
#undef HANDLE_ERROR_STMT
}

