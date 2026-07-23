"""
Correctness tests for the Triton RMSNorm kernel — Phase 2.

All tests in this file require a CUDA GPU and triton>=2.3.  They are
automatically skipped when either is absent.

Test strategy
─────────────
Every test compares the Triton output against the FP32 PyTorch reference
using the same per-dtype tolerances defined in tolerances.py.  This is the
same contract used in Phase 3 for the CUDA C++ kernel.

Correctness is validated before any performance claim is made.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Skip the entire module if triton or CUDA is not available.
triton = pytest.importorskip("triton", reason="triton not installed")

from gpuforge.ops.reference.rmsnorm import rmsnorm_forward
from gpuforge.ops.reference.tolerances import get_tolerance
from gpuforge.ops.triton.rmsnorm import rmsnorm_triton

pytestmark = pytest.mark.gpu  # every test here requires a CUDA GPU


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_close_to_reference(
    actual: torch.Tensor,
    x_fp32: torch.Tensor,
    weight_fp32: torch.Tensor,
    dtype: torch.dtype,
    *,
    label: str = "",
) -> None:
    """Compare *actual* (Triton output) against the FP32 reference."""
    tol = get_tolerance(dtype)
    ref = rmsnorm_forward(x_fp32, weight_fp32)

    assert not torch.isnan(actual).any(),  f"NaN in Triton output {label}"
    assert not torch.isinf(actual).any(),  f"Inf in Triton output {label}"

    torch.testing.assert_close(
        actual.float(),
        ref,
        atol=tol.atol,
        rtol=tol.rtol,
        msg=(
            f"Triton output exceeds tolerance for {label}: "
            f"atol={tol.atol}, rtol={tol.rtol}"
        ),
    )


# ── 1. dtype correctness ─────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_rmsnorm_dtype_correctness(dtype: torch.dtype) -> None:
    """Triton output must match the FP32 reference within per-dtype tolerance."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    x_fp32 = torch.randn(4, 256, device=device)
    w_fp32 = torch.ones(256, device=device)

    x = x_fp32.to(dtype)
    w = w_fp32.to(dtype)
    out = rmsnorm_triton(x, w)

    assert out.dtype == dtype, f"Expected output dtype {dtype}, got {out.dtype}"
    _assert_close_to_reference(out, x_fp32, w_fp32, dtype, label=f"dtype={dtype}")


# ── 2. shape tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "shape",
    [
        (64,),
        (4, 64),
        (2, 16, 64),
        (2, 4, 8, 64),
        (1, 1, 7),
        (1, 1, 511),
        (1, 1, 512),
        (1, 1, 1024),
        (1, 1, 4096),
        (8, 128, 1024),
    ],
)
def test_triton_rmsnorm_shapes(shape: tuple[int, ...]) -> None:
    """Triton kernel must produce correct output for all supported shapes."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    x_fp32 = torch.randn(*shape, device=device)
    w_fp32 = torch.ones(shape[-1], device=device)

    out = rmsnorm_triton(x_fp32, w_fp32)

    assert out.shape == x_fp32.shape
    _assert_close_to_reference(out, x_fp32, w_fp32, torch.float32, label=f"shape={shape}")


# ── 3. non-contiguous inputs ──────────────────────────────────────────────────

def test_triton_rmsnorm_noncontiguous_input() -> None:
    """Triton kernel must handle non-contiguous inputs via the wrapper's .contiguous() call."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    x = torch.randn(4, 16, 128, device=device)
    x_noncontig = x.permute(1, 0, 2)  # (16, 4, 128), non-contiguous
    assert not x_noncontig.is_contiguous()

    weight = torch.ones(128, device=device)
    out = rmsnorm_triton(x_noncontig, weight)

    assert out.shape == x_noncontig.shape
    ref = rmsnorm_forward(x_noncontig.float(), weight.float())
    torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)


def test_triton_rmsnorm_strided_input() -> None:
    """Strided slice (every other row) must match the contiguous reference."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    x = torch.randn(8, 256, device=device)
    x_strided = x[::2]  # rows 0, 2, 4, 6

    weight = torch.ones(256, device=device)
    out = rmsnorm_triton(x_strided, weight)

    ref = rmsnorm_forward(x_strided.float(), weight.float())
    torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)


# ── 4. fp16 large-value overflow guard ───────────────────────────────────────

def test_triton_rmsnorm_fp16_no_overflow() -> None:
    """fp16 inputs near |x|=200 must not overflow during squaring.

    Without fp32 upcasting: 200² = 40000 which overflows fp16 (max ≈ 65504
    and 200² stored as fp16 is representable, BUT accumulated sum across 256
    elements would be 256 × 40000 = 10.2 million, which overflows fp16).
    The kernel upcasts to fp32 before squaring, preventing this.
    """
    device = torch.device("cuda")
    x = torch.full((2, 256), 200.0, dtype=torch.float16, device=device)
    weight = torch.ones(256, dtype=torch.float16, device=device)

    out = rmsnorm_triton(x, weight)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # All-equal input → RMSNorm output should be ~1.0 everywhere
    torch.testing.assert_close(
        out.float(), torch.ones_like(out.float()), atol=1e-2, rtol=1e-2
    )


# ── 5. edge cases ─────────────────────────────────────────────────────────────

def test_triton_rmsnorm_all_zeros() -> None:
    """All-zero input: eps prevents 0/0; output must be zeros."""
    device = torch.device("cuda")
    x = torch.zeros(4, 128, device=device)
    weight = torch.ones(128, device=device)
    out = rmsnorm_triton(x, weight)
    assert torch.all(out == 0.0)


def test_triton_rmsnorm_unit_weight_unit_rms() -> None:
    """With unit weight and typical N(0,1) inputs, output RMS ≈ 1."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    x = torch.randn(8, 512, device=device)
    weight = torch.ones(512, device=device)
    out = rmsnorm_triton(x, weight)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(8, device=device), atol=1e-3, rtol=1e-3)


# ── 6. invalid-input validation ───────────────────────────────────────────────

def test_triton_rmsnorm_wrong_weight_shape_raises() -> None:
    device = torch.device("cuda")
    x = torch.randn(4, 64, device=device)
    weight = torch.ones(32, device=device)  # wrong dim
    with pytest.raises(ValueError, match="weight dimension"):
        rmsnorm_triton(x, weight)


def test_triton_rmsnorm_negative_eps_raises() -> None:
    device = torch.device("cuda")
    x = torch.randn(4, 64, device=device)
    weight = torch.ones(64, device=device)
    with pytest.raises(ValueError, match="eps must be positive"):
        rmsnorm_triton(x, weight, eps=-1e-6)


# ── 7. agreement with reference across random seeds ──────────────────────────

@pytest.mark.parametrize("seed", range(10))
def test_triton_rmsnorm_matches_reference_random_seeds(seed: int) -> None:
    """Triton must match the FP32 reference for 10 independent random inputs."""
    torch.manual_seed(seed * 100)
    device = torch.device("cuda")
    batch = torch.randint(1, 8, (1,)).item()
    hidden = int(torch.randint(1, 16, (1,)).item()) * 64  # multiples of 64

    x_fp32 = torch.randn(batch, hidden, device=device)
    w_fp32 = torch.randn(hidden, device=device)

    out_triton = rmsnorm_triton(x_fp32, w_fp32)
    out_ref    = rmsnorm_forward(x_fp32, w_fp32)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)


# ── 8. property-based tests (GPU) ────────────────────────────────────────────

@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=256),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_triton_matches_reference(batch: int, hidden: int) -> None:
    """Triton output matches FP32 reference for random (batch, hidden) pairs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(batch * 1000 + hidden)
    device = torch.device("cuda")
    x = torch.randn(batch, hidden, device=device)
    weight = torch.ones(hidden, device=device)

    out_triton = rmsnorm_triton(x, weight)
    out_ref    = rmsnorm_forward(x, weight)

    torch.testing.assert_close(out_triton, out_ref, atol=1e-5, rtol=1e-5)
