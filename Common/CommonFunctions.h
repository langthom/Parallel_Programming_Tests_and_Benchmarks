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
std::vector< int > computeMaskOffsets(int envSize, std::size_t dimY, std::size_t dimX); 


#endif // Parallel_Programming_Tests__CommonFunctions__H