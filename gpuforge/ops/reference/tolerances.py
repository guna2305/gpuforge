"""
Numerical tolerances for comparing floating-point implementations.

These tolerances apply when comparing any implementation against the FP32
PyTorch reference.  They account for:

  * Limited mantissa precision of each dtype.
  * Round-trip quantization (fp32 input → low-precision → fp32 reference).
  * Accumulated rounding during the reduction.

Mantissa bits and machine epsilon by dtype
──────────────────────────────────────────
  float32  23 explicit bits  ε ≈ 1.19 × 10⁻⁷
  float16  10 explicit bits  ε ≈ 9.77 × 10⁻⁴
  bfloat16  7 explicit bits  ε ≈ 7.81 × 10⁻³

The tolerances below are intentionally conservative so the test suite
remains green across different hardware, driver versions, and random seeds.
Tighter bounds are used in individual tests that fix the input.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    atol: float  # absolute tolerance
    rtol: float  # relative tolerance


# fmt: off
DTYPE_TOLERANCES: dict[torch.dtype, Tolerance] = {
    # FP32: comparing our reference to itself — nearly bit-exact, small atol
    # to absorb any instruction reordering differences.
    torch.float32:  Tolerance(atol=1e-5,  rtol=1e-5),

    # FP16: 10-bit mantissa; round-trip from fp32 adds up to ~0.1% error,
    # output quantization adds another ~0.1%, so 1e-2 absolute is safe.
    torch.float16:  Tolerance(atol=1e-2,  rtol=1e-2),

    # BF16: 7-bit mantissa; round-trip adds up to ~0.8% error per element.
    # After RMSNorm outputs land in [-3, 3], 5e-2 absolute covers the worst case.
    torch.bfloat16: Tolerance(atol=5e-2,  rtol=5e-2),
}
# fmt: on


def get_tolerance(dtype: torch.dtype) -> Tolerance:
    """Return the correctness tolerance for the given dtype.

    Raises
    ------
    ValueError
        If no tolerance entry exists for *dtype*.
    """
    if dtype not in DTYPE_TOLERANCES:
        raise ValueError(
            f"No tolerance defined for dtype {dtype}. "
            f"Supported: {list(DTYPE_TOLERANCES.keys())}"
        )
    return DTYPE_TOLERANCES[dtype]
