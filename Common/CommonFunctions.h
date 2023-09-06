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

#ifndef Parallel_Programming_Tests__CommonFunctions__H
#define Parallel_Programming_Tests__CommonFunctions__H

#include <string>
#include <vector>

using int64_t = long long;

/// <summary>
/// Computes the CPU version of the kernel, which computes the first four central statistical moments.
/// </summary>
/// <param name="buffer">The buffer containing the input data.</param>
/// <param name="offsets">The offsets for gathering the voxel values in an environment.</param>
/// <param name="index">The linearized index to the center of the environment.</param>
/// <param name="K">The environment side length.</param>
/// <returns>Returns the first four centralized statistical moments in the indicated environment.</returns>
std::array< float, 4 > stats(float* buffer, int64_t* offsets, int64_t index, int K);

/// <summary>
/// Applies the CPU kernel over the dataset, possibly parallel.
/// </summary>
/// <param name="groundTruth">Pointer to the output buffer, must be properly allocated.</param>
/// <param name="dataPtr">Pointer to the input data.</param>
/// <param name="offsets">The computed offsets for jumping.</param>
/// <param name="N">The total number of voxels in the input volume.</param>
/// <param name="K">The environment side length.</param>
/// <param name="dimX">The tertiary dimension of the volume.</param>
/// <param name="dimY">The secondary dimension of the volume.</param>
/// <param name="dimZ">The primary dimension of the volume.</param>
/// <param name="parallel">Indicator if the execution should be parallel (via OpenMP) or not.</param>
/// <returns>Returns the elapsed time in milliseconds.</returns>
double cpuKernel(float* groundTruth, float* dataPtr, int64_t* offsets, int64_t N, int K, int64_t dimX, int64_t dimY, int64_t dimZ, bool parallel);

/// <summary>
/// Computes the maximum absolute pointwise error between the \a groundTruth and the \a computedOutput.
/// </summary>
/// <param name="groundTruth">The ground truth to compare against.</param>
/// <param name="computedOutput">The computed output to compare.</param>
/// <param name="bufferSize">The size of both buffers in elements/voxels.</param>
/// <returns>Returns the maximum absolute poitnwise error.</returns>
float computeMaxError(float* groundTruth, float* computedOutput, int64_t bufferSize);

/// <summary>
/// Formats a memory specification from bytes into a floating point value
/// and an according unit, e.g., it converts (1ull << 30) into  1.0 GiB.
/// Knowing the limitations of common GPU architectures, we limit ourselves 
/// here only up to GiB, i.e., all units above 1024 GiB will be returned
/// in terms of GiB units.
/// </summary>
/// <param name="memory">The memory in bytes to convert, conveniently already as double precision floating point.</param>
/// <param name="unit">The unit (up to "GiB") that will be assigned to the returned value.</param>
/// <returns>Returns the converted value expressed in terms of the <emph>unit</emph>.</returns>
double formatMemory(double memory, std::string& unit);

/// <summary>
/// Computes the offsets to access a cuboid mask of side length <tt>envSize</tt>, i.e., a
/// cuboid mask encompassing <tt>envSize * envSize * envSize</tt> voxels. The offsets are calculated
/// respecting the dimensions of the volume in the secondary (<tt>dimY</tt>) and tertiary (<tt>dimX</tt>)
/// dimension, thus one can access the voxel values by iterating over these offsets and doing a linear access:
/// 
/// <code>
///   size_t voxelsInRegion = envSize * envSize * envSize;
///   for(size_t i = 0; i < voxelsInRegion; ++i) {
///     auto currentVoxelValue = inputBuffer[anchor + offsets[i]];
///   }
/// </code>
/// 
/// In this example, the <tt>inputBuffer</tt> is a simple buffer type, i.e., a block in memory that can be
/// accessed in a linear fashion. The <tt>anchor</tt> instead is the linear index identifying the center of
/// the region. The linear access by <tt>offsets[i]</tt> will jump backwards and forwards from this linearized
/// center index in order to access the values in the environment.
/// </summary>
/// <param name="envSize">The side length of the environment, assumed to be odd and bigger than one.</param>
/// <param name="dimY">The secondary dimension of the linearized volume.</param>
/// <param name="dimX">The tertiary dimension of the linearized volume.</param>
/// <returns>A vector containing the computed offsets.</returns>
std::vector< int64_t > computeMaskOffsets(int envSize, std::size_t dimY, std::size_t dimX); 


#endif // Parallel_Programming_Tests__CommonFunctions__H