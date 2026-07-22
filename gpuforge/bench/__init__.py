"""
Benchmark harness: CUDA-event timing, CPU fallback, statistical aggregation,
correctness checking, and Pydantic result models.
"""

from gpuforge.bench.models import BenchmarkResult, BenchmarkStats, BenchmarkSuite, CorrectnessResult
from gpuforge.bench.harness import BenchmarkHarness
from gpuforge.bench.timer import KernelTimer

__all__ = [
    "BenchmarkHarness",
    "BenchmarkResult",
    "BenchmarkStats",
    "BenchmarkSuite",
    "CorrectnessResult",
    "KernelTimer",
]
