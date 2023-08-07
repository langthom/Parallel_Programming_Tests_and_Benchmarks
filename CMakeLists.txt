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

cmake_minimum_required(VERSION 3.18)

project(PP_Tests_Bench)

# =================================================================================== #
# Find OpencL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
  add_definitions(-DHAS_OPENCL)
  message(STATUS "[technology] Found OpenCL installation; version ${OpenCL_VERSION_STRING}.")
else()
  message(WARNING "[technology] OpenCL not found, OpenCL tests will not be built or run.")
endif()

# Find CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DHAS_CUDA)
  set(CUDA_FOUND ON)
  message(STATUS "[technology] Found CUDA installation; version ${CUDAToolkit_VERSION}.")
else()
  message(WARNING "[technology] No CUDA installation found. Will skip CUDA tests.")
endif()

# Find MPI
find_package(MPI QUIET)
if(MPI_FOUND)
  message(FATAL_ERROR "NOT IMPLEMENTED YET!!!!")
  message(STATUS "[technology] Found MPI installation; version ${MPI_VERSION}.")
else()
  message(WARNING "[technology] No MPI installation found. Will skip MPI tests.")
endif()


# =================================================================================== #
# Process the other CMake files for building our tests and benchmarks
add_subdirectory(MaskBasedProcessing_OpenCL_vs_CUDA)
add_subdirectory(MeasureMaxUsableMemory)
add_subdirectory(CUDABiggerAllocationTest)
