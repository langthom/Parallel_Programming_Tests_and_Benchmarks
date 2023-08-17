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

#ifdef HAS_CUDA

/// <summary>
/// The statistics kernel computing the first four standardized statistical moments in a single local region
/// centered around the voxel indexed by the \a globalIndex and of side length \a envSize.
/// </summary>
/// <param name="features">The buffer where four feature values are written to.</param>
/// <param name="in">The buffer where we read from.</param>
/// <param name="globalIndex">The linearized index pointing to the center of the local region.</param>
/// <param name="offsets">Offsets array specifying how to jump around in the 3D volume.</param>
/// <param name="envSize">The environment side length.</param>
__device__
void stats(float* features, float* in, int globalIndex, int* offsets, int envSize);

/// <summary>
/// </summary>
/// <param name="features">The buffer where four feature values are written to.</param>
/// <param name="in">The buffer where we read from.</param>
/// <param name="out">The buffere where we write to. Must be properly allocated and of the same size as the input buffer.</param>
/// <param name="offsets">Offsets array specifying how to jump around in the 3D volume.</param>
/// <param name="K">The environment side length.</param>
/// <param name="N">The number of voxels of the volume.</param>
/// <param name="dimX">The tertiary volume dimension.</param>
/// <param name="dimY">The secondary volume dimension.</param>
/// <param name="dimZ">The primary volume dimension.</param>
__global__
void statisticsKernel(float* in, float* out, int* offsets, int const K, size_t N, size_t dimX, size_t dimY, size_t dimZ);

#endif // HAS_CUDA
