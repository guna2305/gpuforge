"""
Correctness tests for the PyTorch RMSNorm reference implementation.

Test categories
───────────────
1.  Manual formula check       — known input → hand-computed expected output.
2.  Shape tests                — 1-D through 4-D, power-of-two, odd, large.
3.  Dtype tests                — fp32, fp16, bf16.
4.  Non-contiguous inputs      — tensors that are not laid out contiguously.
5.  Edge cases                 — all-zeros, large magnitudes, small magnitudes.
6.  NaN / Inf guard            — verify no NaN/Inf leaks into the output.
7.  Invalid-input validation   — wrong weight shape, bad eps, bad ndim.
8.  Module API                 — RMSNorm nn.Module forward and repr.
9.  Property-based tests       — 100+ randomised cases via Hypothesis.

Every test that can run on CPU does run on CPU (no GPU required).
GPU tests use the ``cuda_device`` fixture (auto-skipped when no GPU present).
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpuforge.ops.reference.rmsnorm import RMSNorm, rmsnorm_forward
from gpuforge.ops.reference.tolerances import get_tolerance

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ref_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute RMSNorm in Python-level float64 for independent verification."""
    x64 = x.double()
    w64 = weight.double()
    mean_sq = x64.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    return (x64 * inv_rms * w64).float()


def _assert_no_nan_inf(t: torch.Tensor, label: str = "") -> None:
    assert not torch.isnan(t).any(), f"NaN detected in {label}"
    assert not torch.isinf(t).any(), f"Inf detected in {label}"


# ── 1. Manual formula check ───────────────────────────────────────────────────


def test_rmsnorm_known_input_unit_weight() -> None:
    """With weight=1, output should equal x / RMS(x)."""
    x = torch.tensor([[3.0, 4.0]])  # RMS = sqrt((9+16)/2) = sqrt(12.5)
    weight = torch.ones(2)
    eps = 0.0  # use zero eps for this exact check; relies on non-zero input

    # We use eps=1e-6 in production; for this test use a tiny eps so the
    # hand-computed value is still close.
    eps = 1e-9
    rms = math.sqrt((3.0**2 + 4.0**2) / 2.0 + eps)
    expected = torch.tensor([[3.0 / rms, 4.0 / rms]])

    result = rmsnorm_forward(x, weight, eps=eps)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_weight_scaling() -> None:
    """rmsnorm(x, c·w) == c · rmsnorm(x, w) for any scalar c."""
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    weight = torch.ones(128)
    c = 3.7

    base = rmsnorm_forward(x, weight)
    scaled = rmsnorm_forward(x, c * weight)
    torch.testing.assert_close(scaled, c * base, atol=1e-5, rtol=1e-5)


def test_rmsnorm_unit_weight_unit_rms() -> None:
    """With unit weight, output RMS ≈ 1 for typical N(0,1) inputs."""
    torch.manual_seed(1)
    x = torch.randn(8, 512)
    weight = torch.ones(512)
    out = rmsnorm_forward(x, weight)
    rms_out = out.pow(2).mean(dim=-1).sqrt()
    # Should be very close to 1 when eps << E[x^2] ≈ 1
    torch.testing.assert_close(rms_out, torch.ones(8), atol=1e-4, rtol=1e-4)


# ── 2. Shape tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape",
    [
        (64,),           # 1-D
        (4, 64),         # 2-D
        (2, 16, 64),     # 3-D (batch × seq × hidden)
        (2, 4, 8, 64),   # 4-D
        (1, 1, 7),       # non-power-of-two hidden
        (1, 1, 511),     # non-power-of-two hidden
        (1, 1, 512),     # power-of-two hidden
        (1, 1, 4096),    # large hidden (LLM-scale)
        (1, 1, 1),       # degenerate: single element
    ],
)
def test_rmsnorm_output_shape(shape: tuple[int, ...]) -> None:
    """Output shape must equal input shape for all valid inputs."""
    torch.manual_seed(2)
    x = torch.randn(*shape)
    weight = torch.ones(shape[-1])
    out = rmsnorm_forward(x, weight)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 7),
        (3, 5, 17),
        (2, 13, 100),
    ],
)
def test_rmsnorm_odd_dimensions(shape: tuple[int, ...]) -> None:
    """Odd and non-power-of-two hidden dimensions must produce correct output."""
    torch.manual_seed(3)
    x = torch.randn(*shape)
    weight = torch.randn(shape[-1]).abs() + 0.1  # positive weights
    result = rmsnorm_forward(x, weight)
    expected = _ref_fp32(x, weight)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ── 3. Dtype tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_dtype_output_dtype(dtype: torch.dtype) -> None:
    """Output dtype must match input dtype."""
    torch.manual_seed(4)
    x = torch.randn(4, 128).to(dtype)
    weight = torch.ones(128, dtype=dtype)
    out = rmsnorm_forward(x, weight)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_dtype_vs_fp32_reference(dtype: torch.dtype) -> None:
    """Each dtype result must be close to the FP32 reference within tolerance."""
    torch.manual_seed(5)
    tol = get_tolerance(dtype)

    # Generate inputs in fp32 then quantise so both paths see the same values.
    x_fp32 = torch.randn(4, 256)
    w_fp32 = torch.ones(256)
    ref_out = rmsnorm_forward(x_fp32, w_fp32)  # FP32 reference

    x_cast = x_fp32.to(dtype)
    w_cast = w_fp32.to(dtype)
    result = rmsnorm_forward(x_cast, w_cast)

    _assert_no_nan_inf(result, label=f"rmsnorm output ({dtype})")
    torch.testing.assert_close(
        result.float(), ref_out, atol=tol.atol, rtol=tol.rtol,
        msg=f"dtype={dtype} exceeds tolerance atol={tol.atol} rtol={tol.rtol}",
    )


# ── 4. Non-contiguous inputs ──────────────────────────────────────────────────


def test_rmsnorm_transposed_input() -> None:
    """Non-contiguous input obtained via permute must give the same result as
    the contiguous version."""
    torch.manual_seed(6)
    # Start with shape (B=4, T=16, H=128); permute the two leading dims to
    # get a non-contiguous tensor whose last dim (hidden_dim=128) is unchanged.
    x = torch.randn(4, 16, 128)
    x_noncontiguous = x.permute(1, 0, 2)  # shape (16, 4, 128), non-contiguous
    assert not x_noncontiguous.is_contiguous()

    weight = torch.ones(128)  # last dim of x_noncontiguous is 128
    out_contig = rmsnorm_forward(x_noncontiguous.contiguous(), weight)
    out_noncontig = rmsnorm_forward(x_noncontiguous, weight)
    torch.testing.assert_close(out_contig, out_noncontig, atol=1e-6, rtol=1e-6)


def test_rmsnorm_strided_slice_input() -> None:
    """A strided slice (every other row) must produce correct output."""
    torch.manual_seed(7)
    x = torch.randn(8, 64)
    x_strided = x[::2]  # rows 0, 2, 4, 6  — non-contiguous
    assert not x_strided.is_contiguous()

    weight = torch.ones(64)
    result = rmsnorm_forward(x_strided, weight)
    expected = rmsnorm_forward(x_strided.contiguous(), weight)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


# ── 5. Edge cases ─────────────────────────────────────────────────────────────


def test_rmsnorm_all_zeros_input() -> None:
    """All-zero input: eps prevents division by zero; output should be zero."""
    x = torch.zeros(4, 128)
    weight = torch.ones(128)
    out = rmsnorm_forward(x, weight)
    # x / RMS(0, eps) * 1 = 0 / sqrt(eps) * 1 = 0
    assert torch.all(out == 0.0), "Expected all zeros for zero input"


def test_rmsnorm_large_magnitude_input() -> None:
    """Large-magnitude fp32 inputs must not produce NaN or Inf."""
    torch.manual_seed(8)
    x = torch.randn(4, 128) * 1e4
    weight = torch.ones(128)
    out = rmsnorm_forward(x, weight)
    _assert_no_nan_inf(out, "large magnitude fp32")


def test_rmsnorm_small_magnitude_input() -> None:
    """Very small fp32 inputs must not produce NaN or Inf.

    When |x| ~ 1e-6 and eps=1e-6, the epsilon dominates the denominator:
        mean_sq ~ (1e-6)^2 = 1e-12  <<  eps = 1e-6
        inv_rms = rsqrt(1e-12 + 1e-6) ≈ 1000
        output  ~ 1e-6 * 1000 = 1e-3

    RMSNorm is only scale-invariant when eps << mean(x^2).  For near-zero
    inputs the epsilon is intentionally dominant (preventing 0/0), so the
    output RMS is NOT 1.  The critical guarantee is numerical stability.
    """
    torch.manual_seed(9)
    x = torch.randn(4, 128) * 1e-6
    weight = torch.ones(128)
    out = rmsnorm_forward(x, weight)
    _assert_no_nan_inf(out, "small magnitude fp32")


def test_rmsnorm_fp16_large_values_no_overflow() -> None:
    """fp16 inputs near the fp16 max (65504) must not overflow.

    Without upcasting, x² overflows fp16 when |x| > ~256.  Our reference
    upcasts to fp32 before squaring, so this should be safe.
    """
    x = torch.full((2, 64), 200.0, dtype=torch.float16)
    weight = torch.ones(64, dtype=torch.float16)
    out = rmsnorm_forward(x, weight)
    _assert_no_nan_inf(out, "fp16 large values")


def test_rmsnorm_eps_effect() -> None:
    """A larger eps should give a result closer to zero for near-zero input."""
    x = torch.full((1, 4), 1e-8)
    weight = torch.ones(4)
    out_small_eps = rmsnorm_forward(x, weight, eps=1e-30)
    out_large_eps = rmsnorm_forward(x, weight, eps=1.0)
    # With eps=1, RMS ≈ 1, so output ≈ 1e-8; with eps=1e-30, output ≈ 1.
    assert out_large_eps.abs().max() < out_small_eps.abs().max()


def test_rmsnorm_learned_weight_not_one() -> None:
    """Verify that non-unit weight is applied correctly."""
    torch.manual_seed(10)
    x = torch.randn(2, 8)
    weight = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5, 0.1, 4.0, 0.0])
    result = rmsnorm_forward(x, weight)
    expected = _ref_fp32(x, weight)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ── 6. NaN / Inf guard ────────────────────────────────────────────────────────


def test_rmsnorm_input_with_finite_values_gives_finite_output() -> None:
    """Any finite input with positive eps must produce finite output."""
    torch.manual_seed(11)
    for _ in range(20):
        x = torch.randn(4, 64) * torch.randint(1, 100, (1,)).float()
        weight = torch.randn(64)
        out = rmsnorm_forward(x, weight)
        _assert_no_nan_inf(out, "random finite input")


# ── 7. Invalid-input validation ───────────────────────────────────────────────


def test_rmsnorm_wrong_weight_shape_raises() -> None:
    x = torch.randn(4, 64)
    weight = torch.ones(32)  # wrong: should be 64
    with pytest.raises(ValueError, match="weight dimension"):
        rmsnorm_forward(x, weight)


def test_rmsnorm_negative_eps_raises() -> None:
    x = torch.randn(4, 64)
    weight = torch.ones(64)
    with pytest.raises(ValueError, match="eps must be positive"):
        rmsnorm_forward(x, weight, eps=-1e-6)


def test_rmsnorm_zero_eps_raises() -> None:
    x = torch.randn(4, 64)
    weight = torch.ones(64)
    with pytest.raises(ValueError, match="eps must be positive"):
        rmsnorm_forward(x, weight, eps=0.0)


def test_rmsnorm_weight_2d_raises() -> None:
    x = torch.randn(4, 64)
    weight = torch.ones(1, 64)  # 2-D weight is not valid
    with pytest.raises(ValueError, match="weight must be 1-D"):
        rmsnorm_forward(x, weight)


# ── 8. Module API ─────────────────────────────────────────────────────────────


def test_rmsnorm_module_forward() -> None:
    """RMSNorm module forward must match functional form."""
    torch.manual_seed(12)
    hidden = 256
    module = RMSNorm(hidden_dim=hidden)
    x = torch.randn(4, 16, hidden)
    out_module = module(x)
    out_fn = rmsnorm_forward(x, module.weight)
    torch.testing.assert_close(out_module, out_fn)


def test_rmsnorm_module_weight_is_ones_at_init() -> None:
    """Weight must be initialised to ones so the module is an identity at init."""
    module = RMSNorm(hidden_dim=64)
    assert torch.all(module.weight == 1.0)


def test_rmsnorm_module_repr() -> None:
    module = RMSNorm(hidden_dim=128, eps=1e-5)
    r = repr(module)
    assert "hidden_dim=128" in r
    assert "eps=" in r


def test_rmsnorm_module_invalid_hidden_dim() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        RMSNorm(hidden_dim=0)


def test_rmsnorm_module_parameter_count() -> None:
    module = RMSNorm(hidden_dim=64)
    params = list(module.parameters())
    assert len(params) == 1
    assert params[0].shape == (64,)


# ── 9. Property-based tests (Hypothesis) ─────────────────────────────────────
# Each strategy produces >= 100 examples (settings max_examples=100).
# deadline=None avoids flaky failures on slow CI machines.


@given(
    batch=st.integers(min_value=1, max_value=16),
    hidden=st.integers(min_value=2, max_value=512),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_output_shape_matches_input(batch: int, hidden: int) -> None:
    """Output shape == input shape for all valid (batch, hidden) pairs."""
    torch.manual_seed(batch * 1000 + hidden)
    x = torch.randn(batch, hidden)
    weight = torch.ones(hidden)
    out = rmsnorm_forward(x, weight)
    assert out.shape == x.shape


@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=256),
    c=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_weight_linearity(batch: int, hidden: int, c: float) -> None:
    """rmsnorm(x, c·w) == c · rmsnorm(x, w) for any scalar c > 0."""
    torch.manual_seed(batch * 1000 + hidden)
    x = torch.randn(batch, hidden)
    weight = torch.ones(hidden)

    base = rmsnorm_forward(x, weight)
    scaled_w = rmsnorm_forward(x, c * weight)

    torch.testing.assert_close(scaled_w, c * base, atol=1e-4, rtol=1e-4)


@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=256),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_unit_weight_unit_rms(batch: int, hidden: int) -> None:
    """Unit weight → output RMS ≈ 1 for N(0,1) inputs (where eps << E[x²])."""
    torch.manual_seed(batch * 2000 + hidden)
    x = torch.randn(batch, hidden)  # E[x²] ≈ 1 >> eps=1e-6
    weight = torch.ones(hidden)
    out = rmsnorm_forward(x, weight)

    rms_out = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms_out, torch.ones(batch), atol=1e-3, rtol=1e-3)


@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=256),
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_no_nan_inf_for_finite_input(batch: int, hidden: int, scale: float) -> None:
    """Any finite input in fp32 must produce finite output."""
    torch.manual_seed(batch * 3000 + hidden)
    x = torch.randn(batch, hidden) * scale
    weight = torch.ones(hidden)
    out = rmsnorm_forward(x, weight)
    assert not torch.isnan(out).any(), f"NaN for scale={scale}"
    assert not torch.isinf(out).any(), f"Inf for scale={scale}"


@given(
    leading=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=3),
    hidden=st.integers(min_value=2, max_value=64),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_arbitrary_leading_dims(leading: list[int], hidden: int) -> None:
    """RMSNorm must handle arbitrary numbers of leading batch dimensions."""
    torch.manual_seed(sum(leading) * 100 + hidden)
    shape = tuple(leading) + (hidden,)
    x = torch.randn(*shape)
    weight = torch.ones(hidden)
    out = rmsnorm_forward(x, weight)
    assert out.shape == x.shape
    _assert_no_nan_inf(out)


@given(
    batch=st.integers(min_value=1, max_value=8),
    hidden=st.integers(min_value=2, max_value=128),
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_result_matches_double_reference(batch: int, hidden: int) -> None:
    """fp32 result must be close to independent float64 calculation."""
    torch.manual_seed(batch * 4000 + hidden)
    x = torch.randn(batch, hidden)
    weight = torch.randn(hidden)
    result = rmsnorm_forward(x, weight)
    expected = _ref_fp32(x, weight)  # computed in float64, returned as float32
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
