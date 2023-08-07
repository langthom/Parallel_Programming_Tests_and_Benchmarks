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

/// <summary>
/// Computes the maximum memory in GiB that can be used on the device(s). Given an initial
/// <tt>maxMemoryGiB</tt> value, this value is conditionally reduced to accomodate the available
/// GPUs, retrieving - if present - all CUDA and OpenCL capable devices.
/// Finally, since auxiliary memory is necessary during the kernel execution, only a certain 
/// percentage of the final value is used, specified by <tt>actuallyUsePercentage</tt>.
/// </summary>
/// <param name="maxMemoryGiB">The maximum memory in GiB that shall be used if available.</param>
/// <param name="actuallyUsePercentage">Multiplicative factor (percentage in [0,1]) how much actually should be used. Defaults to 90%.</param>
/// <param name="respectOpenCLLimit">Indicator if the OpenCL limit for a single allocation should be respected.</param>
/// <returns>Returns the maximum memory in GiB suitable for execution on the device.</returns>
double getMaxMemoryInGiB(double maxMemoryGiB, double actuallyUsePercentage = 0.90, bool respectOpenCLLimit = true);

/* ================================================  CUDA  stuff ================================================= */
#ifdef HAS_CUDA

/// <summary>
/// Launches the statistical moments kernel (specified in CommonKernels.cu) on a CUDA device.
/// The <tt>threadsPerBlock</tt> parameter will determine the grid size, i.e., the number of blocks
/// by setting it such that the grid covers the entire dataset.
/// </summary>
/// <param name="out">Output buffer where the result is written into. Must be properly allocated and capable of holding at least <tt>N</tt> elements.</param>
/// <param name="in">Input buffer where the data is read from. Must be properly allocated and contain <tt>N</tt> elements.</param>
/// <param name="N">The number of data points to process.</param>
/// <param name="offsets">Jump offsets starting from the center of a local region to enumerate all environment voxels. Must contain <tt>K*K*K</tt> values.</param>
/// <param name="K">The environment side length, i.e., a local environment is of size <tt>K * K * K</tt> voxels.</param>
/// <param name="dimX">The tertiary dimension of the voxel volume.</param>
/// <param name="dimY">The secondary dimension of the voxel volume.</param>
/// <param name="dimZ">The primary dimension of the voxel volume.</param>
/// <param name="elapsedTimeInMilliseconds">Buffer where the elapsed time in milliseconds will be written into.</param>
/// <param name="deviceID">Identifier of the device to launch this kernel.</param>
/// <param name="threadsPerBlock">Number of threads to use per block.</param>
/// <returns>An error code, if any.</returns>
cudaError_t launchKernel(float* out,
                         float* in,
                         size_t N,
                         int* offsets,
                         int K,
                         size_t dimX,
                         size_t dimY,
                         size_t dimZ,
                         float* elapsedTimeInMilliseconds,
                         int deviceID,
                         int threadsPerBlock);

/// <summary>
/// Retrieves information about all available CUDA devices. All parameters are output parameters.
/// </summary>
/// <param name="nr_gpus">The total number of CUDA compatible devices, ordered by their ID.</param>
/// <param name="deviceNames">The names of all devices, ordered by their ID.</param>
/// <param name="availableMemoryPerDevice">The available (free) GPU (global) memory in bytes per device, ordered by the device ID.</param>
/// <param name="totalMemoryPerDevice">The total (global) memory in bytes per device, ordered by the device ID.</param>
/// <returns>Returns an error code, if any.</returns>
cudaError_t getGPUInformation(int& nr_gpus, 
                              std::vector< std::string >& deviceNames,
                              std::vector< size_t >& availableMemoryPerDevice,
                              std::vector< size_t >& totalMemoryPerDevice);

/// <summary>
/// Retrieves the maximum potential block size available through the CUDA runtime 
/// for a CUDA device identified by <tt>deviceID</tt>.
/// </summary>
/// <param name="maxPotentialBlockSize">Output reference where the max. potential block size will be written to.</param>
/// <param name="deviceID">Identifier identifying the CUDA device.</param>
/// <returns>Returns an error value, if any.</returns>
cudaError_t getMaxPotentialBlockSize(int& maxPotentialBlockSize, int deviceID);

#endif // HAS_CUDA

/* ================================================ OpenCL stuff ================================================= */
#ifdef HAS_OPENCL

/// <summary>
/// Returns the OpenCL kernel which computes the first four central normalized statistical moments
/// and assigns the mean value of a voxel region to each voxel. In the padding region, a value of zero 
/// will be assigned to the respective voxels.
/// </summary>
/// <returns>Returns the kernel as a string.</returns>
std::string getOpenCLKernel();

/// <summary>
/// Returns an error message printing details associated to the passed <tt>errorCode</tt> which
/// occured at the specified <tt>line</tt> in the code.
/// </summary>
/// <param name="errorCode">The OpenCL error code to translate.</param>
/// <param name="line">The line number in the code where the error occured.</param>
/// <returns>Returns the error message.</returns>
/// <throws>Throws an exception of the given <tt>errorCode</tt> is not recognized.</throws>
std::string getOpenCLError(cl_int errorCode, int line);

#endif // HAS_OPENCL

#endif // Parallel_Programming_Tests__CommonKernels__H