#!/usr/bin/env python3
"""
RMSNorm reference benchmark — Phase 1.

Measures the latency and effective memory bandwidth of the pure-PyTorch
RMSNorm reference implementation across a range of shapes and dtypes.

This script is the template for the benchmark framework.  Later phases add
Triton and CUDA implementations alongside the reference so speedup tables
can be generated.

Usage
─────
    # CPU (works without a GPU):
    python benchmarks/bench_rmsnorm.py

    # GPU (auto-detected):
    python benchmarks/bench_rmsnorm.py

    # Custom options:
    python benchmarks/bench_rmsnorm.py \\
        --n-iter 200 --n-warmup 50 \\
        --output artifacts/phase1_rmsnorm.json

Output
──────
A console table and (optionally) a JSON file conforming to BenchmarkSuite.
Never hardcode performance numbers in this file; all numbers come from
the actual measured run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running as `python benchmarks/bench_rmsnorm.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpuforge.bench.harness import BenchmarkHarness
from gpuforge.bench.models import BenchmarkSuite
from gpuforge.ops.reference.rmsnorm import rmsnorm_forward
from gpuforge.ops.reference.tolerances import get_tolerance
from gpuforge.utils.gpu_info import get_gpu_info, get_gpu_name

# ── Benchmark matrix ──────────────────────────────────────────────────────────
#
# Shapes reflect typical LLM workloads:
#   (batch, seq_len, hidden_dim)  or  (tokens, hidden_dim)
#
# We include:
#   - Single-token decode (batch=1, seq=1) — the memory-bound autoregressive case
#   - Prefill sequences   (seq>1)          — larger working sets
#   - Power-of-two and non-power-of-two hidden dims
#   - A large hidden to stress L2/DRAM bandwidth

DEFAULT_SHAPES: list[tuple[int, ...]] = [
    # Single-token decode: latency-sensitive, hidden_dim dominates
    (1, 1, 256),
    (1, 1, 512),
    (1, 1, 1024),
    (1, 1, 2048),
    (1, 1, 4096),
    # Prefill: more tokens, larger working set
    (1, 128, 1024),
    (1, 512, 1024),
    (4, 128, 1024),
    (8, 128, 1024),
    # Non-power-of-two (stress-test memory alignment)
    (1, 1, 7),
    (1, 1, 511),
    (3, 17, 100),
]

DEFAULT_DTYPES: list[torch.dtype] = [torch.float32, torch.float16]


# ── Pretty printing ───────────────────────────────────────────────────────────

def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _shape_str(shape: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape)


def _print_header(device: torch.device, gpu_info: dict | None) -> None:
    print()
    print("=" * 72)
    print("  GPUForge Phase 1 - RMSNorm Reference Benchmark")
    print("=" * 72)
    if gpu_info:
        print(f"  GPU  : {gpu_info['name']}")
        print(f"  SM   : {gpu_info['compute_capability']}   "
              f"VRAM: {gpu_info['total_memory_gb']:.1f} GB   "
              f"SMs: {gpu_info['multiprocessor_count']}")
        print(f"  CUDA : {gpu_info['cuda_version']}   "
              f"PyTorch: {gpu_info['torch_version']}")
    else:
        print(f"  Device : CPU (no CUDA GPU detected)")
        print(f"  PyTorch: {torch.__version__}")
    print("-" * 72)
    print(
        f"  {'Shape':<18}  {'dtype':<8}  "
        f"{'Median':>8}  {'p95':>8}  {'BW (GB/s)':>10}  {'CV':>6}  {'OK?':>4}"
    )
    print("-" * 72)


def _print_row(shape: tuple[int, ...], dtype: torch.dtype, result) -> None:
    bw = f"{result.effective_bandwidth_gbs:.1f}" if result.effective_bandwidth_gbs else "  N/A"
    ok = "PASS" if (result.correctness and result.correctness.passed) else "FAIL"
    print(
        f"  {_shape_str(shape):<18}  {_dtype_str(dtype):<8}  "
        f"{result.stats.median_ms:>7.3f}ms  "
        f"{result.stats.p95_ms:>7.3f}ms  "
        f"{bw:>10}  "
        f"{result.stats.cv:>5.1%}  "
        f"{ok:>4}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMSNorm reference benchmark (Phase 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-iter", type=int, default=100,
        help="Number of timed iterations per configuration",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=20,
        help="Number of warm-up iterations (discarded)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (e.g. artifacts/bench.json)",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU even when a CUDA GPU is available",
    )
    args = parser.parse_args()

    # ── device selection ──────────────────────────────────────────────────────
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    gpu_info = get_gpu_info() if device.type == "cuda" else None

    _print_header(device, gpu_info)

    harness = BenchmarkHarness(device, default_n_warmup=args.n_warmup, default_n_iter=args.n_iter)

    suite = BenchmarkSuite(
        gpu_name=get_gpu_name(device),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
    )

    failed: list[str] = []

    for shape in DEFAULT_SHAPES:
        for dtype in DEFAULT_DTYPES:
            hidden_dim = shape[-1]
            tol = get_tolerance(dtype)

            # Create inputs on the target device
            torch.manual_seed(42)
            x = torch.randn(*shape, dtype=dtype, device=device)
            weight = torch.ones(hidden_dim, dtype=dtype, device=device)

            # FP32 reference output (always computed in fp32 regardless of dtype)
            x_fp32 = x.float()
            w_fp32 = weight.float()
            ref_out = rmsnorm_forward(x_fp32, w_fp32)

            fn = lambda x=x, weight=weight: rmsnorm_forward(x, weight)

            result = harness.run(
                operator="rmsnorm",
                implementation="reference_pytorch",
                fn=fn,
                reference_output=ref_out,
                dtype=dtype,
                shape=list(shape),
                tolerance=tol,
            )

            suite.add(result)
            _print_row(shape, dtype, result)

            if result.correctness and not result.correctness.passed:
                failed.append(
                    f"  FAIL  shape={shape}  dtype={dtype}  "
                    f"max_abs={result.correctness.max_abs_error:.2e}"
                )

    print("-" * 72)
    print(f"  {len(suite.results)} configurations benchmarked.")

    if failed:
        print("\n  CORRECTNESS FAILURES:")
        for msg in failed:
            print(msg)
        sys.exit(1)
    else:
        print("  All correctness checks passed.")

    # ── save results ──────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(suite.model_dump(), f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")

    print()


if __name__ == "__main__":
    main()
