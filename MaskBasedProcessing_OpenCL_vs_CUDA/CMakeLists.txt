
find_package(OpenMP)

if(CUDA_FOUND)
  add_executable(MaskBasedProcessingTestCUDA
    MaskBasedProcessing_OpenCL_vs_CUDA_Main.cpp
    MaskBasedProcessing_OpenCL_vs_CUDA.h
    MaskBasedProcessing_OpenCL_vs_CUDA.cu
  )
  target_link_libraries(MaskBasedProcessingTestCUDA CUDA::cudart)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(MaskBasedProcessingTestCUDA OpenMP::OpenMP_CXX)
  endif()
endif()

