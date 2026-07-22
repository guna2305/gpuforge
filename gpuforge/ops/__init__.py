"""
Operator implementations, layered by correctness guarantee and optimization level.

  reference/  Pure PyTorch — the ground truth for all numerical comparisons.
  triton/     Triton GPU kernels (Phase 2+).
  cuda/       Hand-written CUDA C++ kernels (Phase 3+).
"""
