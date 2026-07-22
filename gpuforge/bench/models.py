"""
Pydantic models for benchmark results.

Every benchmark run produces a BenchmarkResult.  A BenchmarkSuite
collects results from a single invocation of the benchmark script.
These models are serialisable to JSON so results can be stored,
compared across runs, and loaded by the Phase 7 dashboard.

Design rules
────────────
* Never store fabricated numbers.  Fields that cannot be computed on the
  current hardware are left as None.
* BenchmarkStats always requires at least 2 samples (so std is defined).
* CorrectnessResult stores the raw error metrics even when the test fails,
  so failures are diagnosable rather than just "FAIL".
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class BenchmarkStats(BaseModel):
    """Statistical summary of a sequence of kernel latency measurements."""

    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float  # 95th-percentile latency
    cv: float  # coefficient of variation = std / mean (dimensionless)
    n_samples: int

    @field_validator("n_samples")
    @classmethod
    def _min_samples(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"n_samples must be >= 2 (got {v}); cannot compute std")
        return v

    @field_validator("median_ms", "mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms")
    @classmethod
    def _finite_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"Latency values must be finite and non-negative, got {v}")
        return v

    @property
    def throughput_relative_to_median(self) -> float:
        """1.0 = median throughput baseline (useful for speedup comparisons)."""
        return 1.0


class CorrectnessResult(BaseModel):
    """Numerical comparison of an implementation against the FP32 reference."""

    passed: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float  # relative to |reference| + 1e-8
    tolerance_atol: float
    tolerance_rtol: float

    @model_validator(mode="after")
    def _non_negative_errors(self) -> "CorrectnessResult":
        for field in ("max_abs_error", "mean_abs_error", "max_rel_error"):
            val = getattr(self, field)
            if val < 0:
                raise ValueError(f"{field} must be non-negative, got {val}")
        return self


class BenchmarkResult(BaseModel):
    """Result of benchmarking a single (operator, implementation, shape, dtype) tuple."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # What was measured
    operator: str  # e.g. "rmsnorm"
    implementation: str  # e.g. "reference_pytorch", "triton", "cuda_v1"
    dtype: str  # "float32", "float16", "bfloat16"
    shape: list[int]  # full input shape, e.g. [4, 512, 1024]

    # Where it was measured
    device: str  # "cuda:0", "cpu"
    gpu_name: Optional[str] = None  # None on CPU

    # Timing
    stats: BenchmarkStats

    # Correctness vs FP32 reference
    correctness: Optional[CorrectnessResult] = None

    # Derived throughput metrics (None when not applicable / not computed)
    effective_bandwidth_gbs: Optional[float] = None  # GB/s of global memory

    notes: Optional[str] = None

    @property
    def hidden_dim(self) -> int:
        return self.shape[-1]

    @property
    def num_elements(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result


class BenchmarkSuite(BaseModel):
    """Collection of BenchmarkResults from a single script invocation."""

    suite_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Environment metadata
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: str

    results: list[BenchmarkResult] = Field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def by_implementation(self, implementation: str) -> list[BenchmarkResult]:
        return [r for r in self.results if r.implementation == implementation]

    def by_operator(self, operator: str) -> list[BenchmarkResult]:
        return [r for r in self.results if r.operator == operator]


# ── Helper: compute BenchmarkStats from a list of latencies ──────────────────

def compute_stats(latencies_ms: list[float]) -> BenchmarkStats:
    """Aggregate a list of per-iteration latency measurements (milliseconds).

    Raises
    ------
    ValueError
        If fewer than 2 measurements are provided.
    """
    if len(latencies_ms) < 2:
        raise ValueError(
            f"Need at least 2 latency samples to compute stats; got {len(latencies_ms)}"
        )
    arr = np.array(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))  # sample std
    cv = std / mean if mean > 0 else 0.0

    return BenchmarkStats(
        median_ms=float(np.median(arr)),
        mean_ms=mean,
        std_ms=std,
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        p95_ms=float(np.percentile(arr, 95)),
        cv=cv,
        n_samples=len(latencies_ms),
    )
