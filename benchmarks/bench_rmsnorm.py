#!/usr/bin/env python3
"""
RMSNorm benchmark — Phase 1 + 2.

Compares:
  reference_pytorch   Pure PyTorch (Phase 1)
  triton              Triton kernel with autotuning (Phase 2)

Usage
─────
    # CPU only (reference only, no GPU required):
    python benchmarks/bench_rmsnorm.py --cpu-only

    # GPU: reference + triton comparison:
    python benchmarks/bench_rmsnorm.py

    # Choose implementations explicitly:
    python benchmarks/bench_rmsnorm.py --implementations reference_pytorch,triton

    # Save JSON results:
    python benchmarks/bench_rmsnorm.py --output artifacts/phase2_rmsnorm.json

    # Export Triton autotune configs:
    python benchmarks/bench_rmsnorm.py --export-autotune

Output
──────
Console table with median latency, p95, effective bandwidth, speedup vs
reference, and (for Triton) the winning autotune config.
A JSON file following the BenchmarkSuite schema is written when --output is set.

Note: All numbers come from actual measured runs. Nothing is fabricated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpuforge.bench.harness import BenchmarkHarness
from gpuforge.bench.models import BenchmarkResult, BenchmarkSuite
from gpuforge.ops.reference.rmsnorm import rmsnorm_forward
from gpuforge.ops.reference.tolerances import get_tolerance
from gpuforge.utils.gpu_info import get_gpu_info, get_gpu_name

# ── Implementation registry ───────────────────────────────────────────────────
#
# Each entry is a factory: given (x, weight) it returns a zero-argument
# callable suitable for the benchmark harness.
# New implementations (CUDA in Phase 3) are added here.

def _make_reference(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    return lambda: rmsnorm_forward(x, w)


def _make_triton(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    from gpuforge.ops.triton.rmsnorm import rmsnorm_triton
    return lambda: rmsnorm_triton(x, w)


IMPLEMENTATION_FACTORIES: dict[str, Callable] = {
    "reference_pytorch": _make_reference,
    "triton":            _make_triton,
}

# ── Default benchmark matrix ──────────────────────────────────────────────────
#
# Shapes reflect common LLM workloads:
#   (batch, seq_len, hidden_dim)
# Include both power-of-two and non-power-of-two hidden dims.

DEFAULT_SHAPES: list[tuple[int, ...]] = [
    (1, 1,   256),
    (1, 1,   512),
    (1, 1,  1024),
    (1, 1,  2048),
    (1, 1,  4096),
    (1, 128, 1024),
    (1, 512, 1024),
    (4, 128, 1024),
    (8, 128, 1024),
    (1, 1,     7),   # non-power-of-two
    (1, 1,   511),
]

DEFAULT_DTYPES: list[torch.dtype] = [torch.float32, torch.float16]

# ── Formatting helpers ────────────────────────────────────────────────────────

def _dtype_tag(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _shape_tag(shape: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in shape)


def _speedup_str(result: BenchmarkResult, reference: Optional[BenchmarkResult]) -> str:
    if reference is None or reference.stats.median_ms == 0:
        return "  ---  "
    ratio = reference.stats.median_ms / result.stats.median_ms
    return f"{ratio:>5.2f}x"


# ── Console output ────────────────────────────────────────────────────────────

def _print_header(device: torch.device, gpu_info: Optional[dict]) -> None:
    print()
    print("=" * 78)
    print("  GPUForge - RMSNorm Benchmark  (Phase 1: reference | Phase 2: Triton)")
    print("=" * 78)
    if gpu_info:
        print(f"  GPU   : {gpu_info['name']}")
        print(f"  SM    : cc{gpu_info['compute_capability']}   "
              f"VRAM: {gpu_info['total_memory_gb']:.1f} GB   "
              f"SMs: {gpu_info['multiprocessor_count']}")
        print(f"  CUDA  : {gpu_info['cuda_version']}   PyTorch: {gpu_info['torch_version']}")
    else:
        print(f"  Device: CPU (no CUDA GPU detected)")
        print(f"  PyTorch: {torch.__version__}")
    print("-" * 78)


def _print_group_header(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    print(f"\n  shape={_shape_tag(shape)}  dtype={_dtype_tag(dtype)}")
    print(f"  {'impl':<22} {'median':>9} {'p95':>9} {'BW(GB/s)':>10} "
          f"{'speedup':>8} {'CV':>5}  {'ok?':>4}")
    print("  " + "-" * 68)


def _print_row(
    impl_name: str,
    result: BenchmarkResult,
    reference: Optional[BenchmarkResult],
    extra: str = "",
) -> None:
    bw   = f"{result.effective_bandwidth_gbs:.1f}" if result.effective_bandwidth_gbs else " N/A"
    spd  = _speedup_str(result, reference)
    ok   = "PASS" if (result.correctness and result.correctness.passed) else "FAIL"
    print(
        f"  {impl_name:<22} "
        f"{result.stats.median_ms:>7.3f}ms "
        f"{result.stats.p95_ms:>7.3f}ms "
        f"{bw:>10} "
        f"{spd:>8} "
        f"{result.stats.cv:>4.1%}  "
        f"{ok:>4}"
        + (f"  {extra}" if extra else "")
    )


# ── Autotune config label ─────────────────────────────────────────────────────

def _triton_config_label(n: int) -> str:
    """Return a short label for the autotune winner at this N value, if known."""
    try:
        from gpuforge.ops.triton.rmsnorm import get_autotune_best_configs
        configs = get_autotune_best_configs()
        # Triton autotune key is a tuple; try both (n,) and str representations
        for key, cfg in configs.items():
            if str(n) in key:
                return f"[BLOCK={cfg['BLOCK_SIZE']}, warps={cfg['num_warps']}]"
    except Exception:
        pass
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RMSNorm multi-implementation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--implementations",
        default="reference_pytorch,triton",
        help="Comma-separated list of implementations to benchmark",
    )
    parser.add_argument("--n-iter",   type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--output",   type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU; only 'reference_pytorch' will run")
    parser.add_argument("--export-autotune", action="store_true",
                        help="Export Triton autotune results to JSON after benchmarking")
    args = parser.parse_args()

    # ── device and implementation selection ───────────────────────────────────
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device("cpu")
        requested = ["reference_pytorch"]
        if args.implementations != "reference_pytorch,triton":
            # User explicitly asked for something other than the default
            requested = [i.strip() for i in args.implementations.split(",")]
            requested = [r for r in requested if r != "triton"]  # silently drop
    else:
        device = torch.device("cuda")
        requested = [i.strip() for i in args.implementations.split(",")]

    # Filter to only available implementations
    available_impls: dict[str, Callable] = {}
    for name in requested:
        if name not in IMPLEMENTATION_FACTORIES:
            print(f"  WARNING: unknown implementation '{name}', skipping")
            continue
        if name == "triton":
            try:
                import triton  # noqa: F401
            except ImportError:
                print("  WARNING: triton not installed — skipping triton implementation")
                print("           Install with: pip install triton>=2.3")
                continue
        available_impls[name] = IMPLEMENTATION_FACTORIES[name]

    if not available_impls:
        print("No implementations to benchmark. Exiting.")
        sys.exit(1)

    gpu_info = get_gpu_info() if device.type == "cuda" else None
    _print_header(device, gpu_info)

    harness = BenchmarkHarness(device, default_n_warmup=args.n_warmup, default_n_iter=args.n_iter)
    suite = BenchmarkSuite(
        gpu_name=get_gpu_name(device),
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
    )

    all_failures: list[str] = []

    for shape in DEFAULT_SHAPES:
        for dtype in DEFAULT_DTYPES:
            N = shape[-1]
            tol = get_tolerance(dtype)

            _print_group_header(shape, dtype)

            # FP32 reference output is computed once and shared across all impls
            torch.manual_seed(42)
            x_fp32 = torch.randn(*shape, device=device)
            w_fp32 = torch.ones(N, device=device)
            ref_out = rmsnorm_forward(x_fp32, w_fp32)  # always fp32

            x = x_fp32.to(dtype)
            w = w_fp32.to(dtype)

            reference_result: Optional[BenchmarkResult] = None

            for impl_name, factory in available_impls.items():
                try:
                    fn = factory(x, w)
                    result = harness.run(
                        operator="rmsnorm",
                        implementation=impl_name,
                        fn=fn,
                        reference_output=ref_out,
                        dtype=dtype,
                        shape=list(shape),
                        tolerance=tol,
                    )
                except Exception as exc:
                    print(f"  {impl_name:<22} ERROR: {exc}")
                    continue

                suite.add(result)

                extra = _triton_config_label(N) if impl_name == "triton" else ""
                _print_row(impl_name, result, reference_result, extra=extra)

                if impl_name == "reference_pytorch":
                    reference_result = result

                if result.correctness and not result.correctness.passed:
                    all_failures.append(
                        f"  FAIL  impl={impl_name}  shape={shape}  dtype={dtype}  "
                        f"max_abs={result.correctness.max_abs_error:.2e}"
                    )

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("-" * 78)
    print(f"  {len(suite.results)} benchmark cases completed.")

    if all_failures:
        print("\n  CORRECTNESS FAILURES:")
        for msg in all_failures:
            print(msg)
        sys.exit(1)
    else:
        print("  All correctness checks passed.")

    # ── save results ──────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(suite.model_dump(), f, indent=2, default=str)
        print(f"\n  Results saved to: {out}")

    # ── export autotune configs ───────────────────────────────────────────────
    if args.export_autotune and "triton" in available_impls:
        try:
            from gpuforge.ops.triton.rmsnorm import export_autotune_results
            configs = export_autotune_results()
            print(f"\n  Autotune configs exported ({len(configs)} N values cached).")
        except Exception as exc:
            print(f"\n  WARNING: could not export autotune results: {exc}")

    print()


if __name__ == "__main__":
    main()
