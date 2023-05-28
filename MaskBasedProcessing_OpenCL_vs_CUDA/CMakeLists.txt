

if(CUDA_FOUND)
  add_executable(MaskBasedProcessingTestCUDA
    MaskBasedProcessing_OpenCL_vs_CUDA_Main.cpp
    MaskBasedProcessing_OpenCL_vs_CUDA.h
    MaskBasedProcessing_OpenCL_vs_CUDA.cu
  )
  target_link_libraries(MaskBasedProcessingTestCUDA CUDA::cudart)
  set_target_properties(MaskBasedProcessingTestCUDA PROPERTIES CUDA_ARCHITECTURES "86")
endif()

