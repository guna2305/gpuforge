# csrc/CMakeLists.txt
#
# Phase 1: skeleton only.  CUDA kernel targets are added in Phase 3.
# Each operator lives in its own subdirectory so it can be compiled and
# tested independently.

if(GPUFORGE_BUILD_CUDA)
  add_subdirectory(ops)
endif()
