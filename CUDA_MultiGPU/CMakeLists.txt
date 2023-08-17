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

if(NOT CUDA_FOUND)
  message(WARNING "[CUDA] CUDA not detected, the multi GPU execution with CUDA will not be built!")
  return()
endif()

set(CUDA_MultiGPU_SRCs
  CUDA_MultiGPU.cpp
  MultiGPUExecution.h
  MultiGPUExecution.cu
  ${Common_SRCs}
)

# Build the executable with all sources that shall be included
add_executable(CUDA_MultiGPU ${CUDA_MultiGPU_SRCs})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(CUDA_MultiGPU OpenMP::OpenMP_CXX)
endif()

if(CUDA_FOUND)
  target_link_libraries(CUDA_MultiGPU CUDA::cudart)
endif()

if(OpenCL_FOUND)
  target_link_libraries(CUDA_MultiGPU ${OpenCL_LIBRARIES})
endif()

