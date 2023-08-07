# MIT License
# 
# Copyright (c) 2023 Dr. Thomas Lang
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(CUDABiggerAllocationTest_SRCs
  CUDABiggerAllocationTest_Main.cpp
  ../Common/CommonKernels.h
  ../Common/CommonFunctions.h
  ../Common/CommonFunctions.cpp
)

if(CUDA_FOUND)
  # While also the .cu file has the preprocessor conditional in it wheter CUDA was found,
  # we include it only conditionally in the CMake as well to avoid any potential problems
  # with CMake not understanding the .cu file extension.
  # Even if it would not be included, the conditional codes in the main file as well
  # as in the header file should ensure that there are no build problems then.
  list(APPEND CUDABiggerAllocationTest_SRCs ../Common/CommonKernels.cu)
endif()


# Build the executable with all sources that shall be included
add_executable(CUDABiggerAllocationTest ${CUDABiggerAllocationTest_SRCs})


if(CUDA_FOUND)
  target_link_libraries(CUDABiggerAllocationTest CUDA::cudart)
endif()

if(OpenCL_FOUND)
  target_link_libraries(CUDABiggerAllocationTest ${OpenCL_LIBRARIES})
endif()

