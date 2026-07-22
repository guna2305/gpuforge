cmake_minimum_required(VERSION 3.18)
project(gpuforge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build options ────────────────────────────────────────────────────────────
# CUDA extensions are disabled until Phase 3 adds the first kernel.
option(GPUFORGE_BUILD_CUDA  "Build CUDA C++ extensions (Phase 3+)" OFF)
option(GPUFORGE_BUILD_TESTS "Build C++ unit tests"                 OFF)

# ─── CUDA language and toolkit ────────────────────────────────────────────────
# Enabled in Phase 3.  Configure target architectures to match your GPU:
#   cmake -DGPUFORGE_BUILD_CUDA=ON \
#         -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
#         -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") ..
if(GPUFORGE_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Default to Ampere + Ada; override on the cmake command line.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
  endif()

  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA toolkit : ${CUDAToolkit_VERSION}")
  message(STATUS "CUDA archs   : ${CMAKE_CUDA_ARCHITECTURES}")

  # PyTorch C++ API — required to build custom ops that integrate with the
  # PyTorch dispatcher.
  find_package(Torch REQUIRED)
  message(STATUS "PyTorch      : ${Torch_VERSION}")
endif()

# ─── Source tree ─────────────────────────────────────────────────────────────
add_subdirectory(csrc)
