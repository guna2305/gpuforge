"""
Pure PyTorch reference implementations.

These are the correctness ground truth used in all numerical comparisons.
Never replace these with optimised versions — they must remain simple and
obviously correct.
"""

from gpuforge.ops.reference.rmsnorm import RMSNorm, rmsnorm_forward
from gpuforge.ops.reference.tolerances import Tolerance, get_tolerance

__all__ = ["RMSNorm", "rmsnorm_forward", "Tolerance", "get_tolerance"]
