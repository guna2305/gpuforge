"""
Triton GPU kernel implementations — Phase 2.

All symbols require a CUDA GPU and triton>=2.3.  Importing this module on a
CPU-only machine is safe; calling the functions raises RuntimeError.
"""

from gpuforge.ops.triton.rmsnorm import (
    export_autotune_results,
    get_autotune_best_configs,
    rmsnorm_triton,
)

__all__ = [
    "rmsnorm_triton",
    "get_autotune_best_configs",
    "export_autotune_results",
]
