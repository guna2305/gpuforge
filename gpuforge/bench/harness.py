"""
BenchmarkHarness: orchestrates timing, correctness checking, and bandwidth
estimation for a single (operator, implementation, shape, dtype) tuple.

Usage pattern
─────────────
    harness = BenchmarkHarness(device)
    fn = lambda: rmsnorm_forward(x, weight)
    result = harness.run(
        operator="rmsnorm",
        implementation="reference_pytorch",
        fn=fn,
        reference_output=ref_out,
        dtype=torch.float32,
        shape=list(x.shape),
        tolerance=get_tolerance(torch.float32),
    )

The harness never modifies the function or its inputs; it is read-only
with respect to the computation being measured.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from gpuforge.bench.models import (
    BenchmarkResult,
    BenchmarkStats,
    CorrectnessResult,
    compute_stats,
)
from gpuforge.bench.timer import KernelTimer
from gpuforge.ops.reference.tolerances import Tolerance
from gpuforge.utils.gpu_info import get_gpu_name


def _element_size(dtype: torch.dtype) -> int:
    """Return the size in bytes of a single element of *dtype*."""
    return torch.tensor([], dtype=dtype).element_size()


def _rmsnorm_bytes(shape: list[int], dtype: torch.dtype) -> int:
    """Estimate the global-memory traffic for one RMSNorm forward pass.

    Traffic model (conservative, ignores L1/L2 hit rate):
        read  x        : prod(shape) * element_size
        read  weight   : shape[-1]   * element_size
        write output   : prod(shape) * element_size
    """
    elem = _element_size(dtype)
    n_elements = math.prod(shape)
    hidden = shape[-1]
    return (2 * n_elements + hidden) * elem


class BenchmarkHarness:
    """Orchestrates timing, correctness, and bandwidth for one benchmark case.

    Parameters
    ----------
    device:
        Target device.  Selects CUDA-event or perf_counter timing.
    default_n_warmup:
        Default warm-up iterations (can be overridden per run).
    default_n_iter:
        Default timed iterations (can be overridden per run).
    """

    def __init__(
        self,
        device: torch.device,
        default_n_warmup: int = 20,
        default_n_iter: int = 100,
    ) -> None:
        self.device = device
        self.timer = KernelTimer(device)
        self.default_n_warmup = default_n_warmup
        self.default_n_iter = default_n_iter

    # ── correctness ──────────────────────────────────────────────────────────

    def check_correctness(
        self,
        actual: torch.Tensor,
        reference: torch.Tensor,
        tolerance: Tolerance,
    ) -> CorrectnessResult:
        """Compare *actual* against *reference* in float32.

        The comparison is always done in float32 regardless of the dtypes
        of the input tensors so that dtype differences don't artificially
        inflate errors.
        """
        actual_f32 = actual.detach().float()
        ref_f32 = reference.detach().float()

        abs_error = (actual_f32 - ref_f32).abs()
        # Avoid division by zero in relative error; eps matches PyTorch allclose.
        rel_error = abs_error / (ref_f32.abs() + 1e-8)

        max_abs = float(abs_error.max())
        mean_abs = float(abs_error.mean())
        max_rel = float(rel_error.max())

        passed = bool(
            torch.allclose(actual_f32, ref_f32, atol=tolerance.atol, rtol=tolerance.rtol)
        )

        return CorrectnessResult(
            passed=passed,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_rel_error=max_rel,
            tolerance_atol=tolerance.atol,
            tolerance_rtol=tolerance.rtol,
        )

    # ── timing ───────────────────────────────────────────────────────────────

    def measure_stats(
        self,
        fn: Callable[[], object],
        n_warmup: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> BenchmarkStats:
        """Return timing statistics for *fn*."""
        latencies = self.timer.measure_latencies_ms(
            fn,
            n_warmup=n_warmup if n_warmup is not None else self.default_n_warmup,
            n_iter=n_iter if n_iter is not None else self.default_n_iter,
        )
        return compute_stats(latencies)

    # ── combined run ─────────────────────────────────────────────────────────

    def run(
        self,
        operator: str,
        implementation: str,
        fn: Callable[[], torch.Tensor],
        reference_output: torch.Tensor,
        dtype: torch.dtype,
        shape: list[int],
        tolerance: Tolerance,
        n_warmup: Optional[int] = None,
        n_iter: Optional[int] = None,
        bandwidth_bytes: Optional[int] = None,
    ) -> BenchmarkResult:
        """Time *fn*, check correctness, and return a BenchmarkResult.

        Parameters
        ----------
        operator:
            Canonical operator name, e.g. ``"rmsnorm"``.
        implementation:
            Implementation tag, e.g. ``"reference_pytorch"``, ``"triton"``.
        fn:
            Zero-argument callable that performs one forward pass and
            returns a tensor (the output to compare for correctness).
        reference_output:
            FP32 reference tensor to compare against.
        dtype:
            Dtype of the input tensor (used to look up tolerances and
            to convert the dtype name to a string).
        shape:
            Full input shape as a Python list.
        tolerance:
            Absolute and relative tolerance for the correctness check.
        n_warmup / n_iter:
            Override the harness defaults.
        bandwidth_bytes:
            Estimated bytes of global-memory traffic per call.  When
            provided, effective_bandwidth_gbs is computed.  When None,
            the harness attempts to derive it using the RMSNorm traffic
            model (read x + weight, write output).
        """
        # Run once to get an output for correctness checking
        with torch.no_grad():
            output = fn()

        correctness = self.check_correctness(output, reference_output, tolerance)

        # Time the function
        stats = self.measure_stats(fn, n_warmup=n_warmup, n_iter=n_iter)

        # Effective memory bandwidth
        bw_bytes = bandwidth_bytes
        if bw_bytes is None and operator == "rmsnorm":
            bw_bytes = _rmsnorm_bytes(shape, dtype)

        bandwidth_gbs: Optional[float] = None
        if bw_bytes is not None and stats.median_ms > 0:
            bandwidth_gbs = (bw_bytes / 1e9) / (stats.median_ms / 1e3)

        return BenchmarkResult(
            operator=operator,
            implementation=implementation,
            dtype=str(dtype).replace("torch.", ""),
            shape=shape,
            device=str(self.device),
            gpu_name=get_gpu_name(self.device),
            stats=stats,
            correctness=correctness,
            effective_bandwidth_gbs=bandwidth_gbs,
        )
