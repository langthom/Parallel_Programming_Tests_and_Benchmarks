
find_package(OpenMP)

set(MaskBasedProcessingTestCUDA_SRCs
  MaskBasedProcessing_OpenCL_vs_CUDA_Main.cpp
  MaskBasedProcessing_OpenCL_vs_CUDA.h
)

if(CUDA_FOUND)
  # While also the .cu file has the preprocessor conditional in it wheter CUDA was found,
  # we include it only conditionally in the CMake as well to avoid any potential problems
  # with CMake not understanding the .cu file extension.
  # Even if it would not be included, the conditional codes in the main file as well
  # as in the header file should ensure that there are no build problems then.
  list(APPEND MaskBasedProcessingTestCUDA_SRCs MaskBasedProcessing_OpenCL_vs_CUDA.cu)
endif()


# Build the executable with all sources that shall be included
add_executable(MaskBasedProcessingTestCUDA ${MaskBasedProcessingTestCUDA_SRCs})


if(OpenMP_CXX_FOUND)
  target_link_libraries(MaskBasedProcessingTestCUDA OpenMP::OpenMP_CXX)
endif()

if(CUDA_FOUND)
  target_link_libraries(MaskBasedProcessingTestCUDA CUDA::cudart)
endif()

if(OpenCL_FOUND)
  target_link_libraries(MaskBasedProcessingTestCUDA ${OpenCL_LIBRARIES})
endif()

