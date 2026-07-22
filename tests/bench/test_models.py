"""
Unit tests for the benchmark Pydantic models and compute_stats helper.

These tests run on CPU with no GPU requirement.
"""

from __future__ import annotations

import pytest
import torch

from gpuforge.bench.models import (
    BenchmarkResult,
    BenchmarkStats,
    BenchmarkSuite,
    CorrectnessResult,
    compute_stats,
)
from gpuforge.bench.timer import KernelTimer
from gpuforge.bench.harness import BenchmarkHarness
from gpuforge.ops.reference.rmsnorm import rmsnorm_forward
from gpuforge.ops.reference.tolerances import get_tolerance


# ── compute_stats ─────────────────────────────────────────────────────────────


def test_compute_stats_basic() -> None:
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_stats(latencies)
    assert stats.n_samples == 5
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(5.0)
    assert stats.median_ms == pytest.approx(3.0)
    assert stats.mean_ms == pytest.approx(3.0)


def test_compute_stats_p95() -> None:
    # 100 samples: 1..100; p95 should be near 95.
    latencies = [float(i) for i in range(1, 101)]
    stats = compute_stats(latencies)
    assert stats.p95_ms >= 94.0
    assert stats.p95_ms <= 96.0


def test_compute_stats_cv() -> None:
    # Identical latencies → cv = 0
    latencies = [5.0] * 10
    stats = compute_stats(latencies)
    assert stats.cv == pytest.approx(0.0, abs=1e-6)


def test_compute_stats_requires_min_two_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compute_stats([1.0])


def test_compute_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_stats([])


# ── BenchmarkStats validation ─────────────────────────────────────────────────


def test_benchmark_stats_rejects_negative_latency() -> None:
    with pytest.raises(Exception):
        BenchmarkStats(
            median_ms=-1.0, mean_ms=1.0, std_ms=0.0,
            min_ms=-1.0, max_ms=1.0, p95_ms=1.0, cv=0.0, n_samples=2,
        )


def test_benchmark_stats_rejects_single_sample() -> None:
    with pytest.raises(Exception):
        BenchmarkStats(
            median_ms=1.0, mean_ms=1.0, std_ms=0.0,
            min_ms=1.0, max_ms=1.0, p95_ms=1.0, cv=0.0, n_samples=1,
        )


# ── CorrectnessResult ─────────────────────────────────────────────────────────


def test_correctness_result_passed() -> None:
    cr = CorrectnessResult(
        passed=True,
        max_abs_error=1e-7,
        mean_abs_error=5e-8,
        max_rel_error=1e-6,
        tolerance_atol=1e-5,
        tolerance_rtol=1e-5,
    )
    assert cr.passed


def test_correctness_result_negative_error_raises() -> None:
    with pytest.raises(Exception):
        CorrectnessResult(
            passed=False,
            max_abs_error=-1.0,
            mean_abs_error=0.0,
            max_rel_error=0.0,
            tolerance_atol=1e-5,
            tolerance_rtol=1e-5,
        )


# ── BenchmarkResult serialisation ─────────────────────────────────────────────


def test_benchmark_result_round_trips_json() -> None:
    stats = compute_stats([float(i) for i in range(1, 11)])
    result = BenchmarkResult(
        operator="rmsnorm",
        implementation="reference_pytorch",
        dtype="float32",
        shape=[4, 128],
        device="cpu",
        stats=stats,
    )
    data = result.model_dump()
    restored = BenchmarkResult.model_validate(data)
    assert restored.operator == result.operator
    assert restored.stats.median_ms == result.stats.median_ms
    assert restored.run_id == result.run_id


# ── BenchmarkSuite ────────────────────────────────────────────────────────────


def test_benchmark_suite_add_and_filter() -> None:
    suite = BenchmarkSuite(torch_version=torch.__version__)
    stats = compute_stats([1.0, 2.0])
    r1 = BenchmarkResult(
        operator="rmsnorm", implementation="ref", dtype="float32",
        shape=[1, 64], device="cpu", stats=stats,
    )
    r2 = BenchmarkResult(
        operator="rmsnorm", implementation="triton", dtype="float16",
        shape=[1, 64], device="cpu", stats=stats,
    )
    suite.add(r1)
    suite.add(r2)
    assert len(suite.results) == 2
    assert len(suite.by_implementation("ref")) == 1
    assert len(suite.by_operator("rmsnorm")) == 2


# ── KernelTimer (CPU path) ────────────────────────────────────────────────────


def test_kernel_timer_cpu_returns_correct_count() -> None:
    device = torch.device("cpu")
    timer = KernelTimer(device)
    fn = lambda: torch.randn(16, 16).sum()
    latencies = timer.measure_latencies_ms(fn, n_warmup=2, n_iter=10)
    assert len(latencies) == 10
    assert all(l > 0 for l in latencies)


def test_kernel_timer_cpu_rejects_single_iter() -> None:
    device = torch.device("cpu")
    timer = KernelTimer(device)
    with pytest.raises(ValueError, match="n_iter must be >= 2"):
        timer.measure_latencies_ms(lambda: None, n_iter=1)


# ── BenchmarkHarness end-to-end (CPU) ────────────────────────────────────────


def test_harness_rmsnorm_reference_passes_correctness() -> None:
    """End-to-end: harness must report correctness=True for the reference vs itself."""
    device = torch.device("cpu")
    harness = BenchmarkHarness(device, default_n_warmup=2, default_n_iter=10)

    torch.manual_seed(42)
    x = torch.randn(2, 64)
    weight = torch.ones(64)
    ref_out = rmsnorm_forward(x.float(), weight.float())
    fn = lambda: rmsnorm_forward(x, weight)
    tol = get_tolerance(torch.float32)

    result = harness.run(
        operator="rmsnorm",
        implementation="reference_pytorch",
        fn=fn,
        reference_output=ref_out,
        dtype=torch.float32,
        shape=list(x.shape),
        tolerance=tol,
    )

    assert result.correctness is not None
    assert result.correctness.passed, (
        f"Reference should be correct vs itself. "
        f"max_abs={result.correctness.max_abs_error:.2e}"
    )
    assert result.effective_bandwidth_gbs is not None
    assert result.effective_bandwidth_gbs > 0


def test_harness_bandwidth_nonzero_on_cpu() -> None:
    """Effective bandwidth is a derived metric; must be positive even on CPU."""
    device = torch.device("cpu")
    harness = BenchmarkHarness(device, default_n_warmup=2, default_n_iter=5)

    x = torch.randn(4, 128)
    weight = torch.ones(128)
    ref_out = rmsnorm_forward(x, weight)
    tol = get_tolerance(torch.float32)

    result = harness.run(
        operator="rmsnorm",
        implementation="reference_pytorch",
        fn=lambda: rmsnorm_forward(x, weight),
        reference_output=ref_out,
        dtype=torch.float32,
        shape=[4, 128],
        tolerance=tol,
    )
    assert result.effective_bandwidth_gbs > 0
