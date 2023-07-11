
set(MeasureMaxUsableMemory_SRCs
  MeasureMaxUsableMemory_Main.cpp
  ../CommonKernels/CommonKernels.h
)

if(CUDA_FOUND)
  # While also the .cu file has the preprocessor conditional in it wheter CUDA was found,
  # we include it only conditionally in the CMake as well to avoid any potential problems
  # with CMake not understanding the .cu file extension.
  # Even if it would not be included, the conditional codes in the main file as well
  # as in the header file should ensure that there are no build problems then.
  list(APPEND MeasureMaxUsableMemory_SRCs ../CommonKernels/CommonKernels.cu)
endif()


# Build the executable with all sources that shall be included
add_executable(MeasureMaxUsableMemory ${MeasureMaxUsableMemory_SRCs})


if(CUDA_FOUND)
  target_link_libraries(MeasureMaxUsableMemory CUDA::cudart)
endif()

if(OpenCL_FOUND)
  target_link_libraries(MeasureMaxUsableMemory ${OpenCL_LIBRARIES})
endif()

