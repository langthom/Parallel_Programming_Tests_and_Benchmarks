cmake_minimum_required(VERSION 3.18)

project(PP_Tests_Bench)

# We require C++17 for aligned storage.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)


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
  set(CUDA_SEPARATE_COMPILATION ON)
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


