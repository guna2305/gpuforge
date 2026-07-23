"""
RMSNorm Triton kernel — Phase 2.

Algorithm
─────────
One Triton program handles one row (one token's hidden vector).
The row is processed in BLOCK_SIZE-wide tiles so that BLOCK_SIZE is a
compile-time constexpr (needed for tl.arange) while N (hidden_dim) is
a runtime value — no upper limit on hidden_dim.

Two-pass layout
───────────────
Pass 1  Load every tile of the row in fp32, accumulate sum(x²).
        After the loop: inv_rms = rsqrt(sum / N + eps).

Pass 2  Load every tile again, apply inv_rms and weight, store in the
        original dtype (fp16 / bf16 / fp32).

Global-memory traffic per element: 2 reads + 1 write = 3 × element_size.
This matches the reference implementation's bandwidth model.

Numerical correctness
─────────────────────
Intermediate reductions are always done in fp32 (matching the reference).
Loading in fp16 then squaring would overflow when |x| > 256 (fp16 max ≈
65504, fp16² max ≈ 4.3 × 10⁹ which overflows).  The fp32 upcast costs
one instruction per element and eliminates that risk.

Autotuning
──────────
@triton.autotune sweeps BLOCK_SIZE ∈ {128, 256, 512, 1024, 2048, 4096}
and num_warps ∈ {2, 4, 8, 16} for each unique N value.  The winner is
cached by Triton in .triton/ and also exportable to JSON via
get_autotune_best_configs().

Design choices vs. the reference
─────────────────────────────────
The reference always uses torch.rsqrt and keeps everything in PyTorch ops.
This Triton kernel:
  • Fuses the two passes into a single kernel launch (less driver overhead).
  • Avoids the intermediate temporary tensor that PyTorch would allocate for
    x_fp32.pow(2).mean(-1, keepdim=True).
  • Allows the autotuner to pick the tile size that best matches the GPU's
    L2 bandwidth and SM occupancy for each hidden_dim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover — only exercised on GPU machines
    _HAS_TRITON = False


# ── Autotune configuration matrix ────────────────────────────────────────────
#
# Rationale for the config range:
#   BLOCK_SIZE < 128  — too small; loop overhead dominates for any real N.
#   BLOCK_SIZE > 4096 — register spill risk; L1 thrashing for large N.
#   num_warps controls threads-per-block = 32 × num_warps.
#   num_stages=2 enables Triton's software pipeliner to overlap the next
#   tl.load with the current computation.

if _HAS_TRITON:
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ]

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N"])
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr,          # [n_rows, N]  contiguous, any real dtype
        W_ptr,          # [N]          contiguous, same dtype as X
        Y_ptr,          # [n_rows, N]  output, same dtype as X
        stride_x_row,   # elements between consecutive rows of X
        N,              # hidden dimension
        eps,            # added inside rsqrt; must be > 0
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(axis=0)

        x_row = X_ptr + row_idx * stride_x_row
        y_row = Y_ptr + row_idx * stride_x_row

        # ── Pass 1: sum(x²) → inv_rms ────────────────────────────────────────
        sum_sq = tl.zeros([1], dtype=tl.float32)

        for start in range(0, N, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            # Upcast to fp32 before squaring to prevent fp16 overflow.
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=0)

        # sum_sq / N  = mean(x²);  +eps prevents 0/0 for all-zero rows
        inv_rms = tl.rsqrt(sum_sq / N + eps)

        # ── Pass 2: normalise and scale ───────────────────────────────────────
        for start in range(0, N, BLOCK_SIZE):
            cols = start + tl.arange(0, BLOCK_SIZE)
            mask = cols < N

            # Load in original dtype so we can cast back correctly at store.
            x_orig = tl.load(x_row + cols, mask=mask, other=0.0)
            w_orig = tl.load(W_ptr  + cols, mask=mask, other=0.0)

            y_fp32 = x_orig.to(tl.float32) * inv_rms * w_orig.to(tl.float32)

            # Cast back to original dtype (fp16 / bf16 / fp32) before storing.
            tl.store(y_row + cols, y_fp32.to(x_orig.dtype), mask=mask)


# ── Python wrapper ────────────────────────────────────────────────────────────

def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Triton-accelerated RMSNorm with hardware-aware autotuning.

    API is intentionally identical to ``rmsnorm_forward`` in the reference
    module so the benchmark harness can swap implementations without changes.

    Parameters
    ----------
    x:
        Input tensor of shape ``(..., hidden_dim)``.  Must be on a CUDA device.
    weight:
        Learned scale of shape ``(hidden_dim,)``.  Same device and dtype as x.
    eps:
        Stability constant.  Default 1e-6 matches the reference.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as x.

    Raises
    ------
    RuntimeError
        If Triton is not installed or no CUDA device is available.
    ValueError
        If weight.shape[0] != x.shape[-1] or eps <= 0.
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "triton is not installed.  Install it with:\n"
            "  pip install triton>=2.3"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Triton kernels require a CUDA GPU.  No CUDA device detected."
        )

    # ── input validation (same contract as reference) ─────────────────────
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {weight.shape}")
    if weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"weight dimension ({weight.shape[0]}) must match "
            f"x last dimension ({x.shape[-1]})"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    original_shape = x.shape
    N = original_shape[-1]

    # Flatten leading dims → 2-D view (n_rows, N); kernel indexes by row
    x_2d = x.contiguous().view(-1, N)
    n_rows = x_2d.shape[0]

    weight_c = weight.contiguous()
    y_2d = torch.empty_like(x_2d)

    # NVTX range for Nsight Systems (Phase 5 profiling)
    torch.cuda.nvtx.range_push("gpuforge::rmsnorm_triton_fwd")
    try:
        _rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            weight_c,
            y_2d,
            x_2d.stride(0),
            N,
            eps,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    return y_2d.view(original_shape)


# ── Autotune result export ────────────────────────────────────────────────────

def get_autotune_best_configs() -> dict[str, Any]:
    """Return the best Triton autotune config for every N value seen so far.

    Results are keyed by N (as a string) and include BLOCK_SIZE, num_warps,
    and num_stages.  Returns an empty dict when Triton is not installed or
    no benchmarks have been run yet.

    These results complement Triton's internal .triton/ cache with our own
    structured JSON format (stored in artifacts/autotune/).
    """
    if not _HAS_TRITON:
        return {}
    try:
        cache = _rmsnorm_fwd_kernel.cache
        return {
            str(key): {
                "BLOCK_SIZE": cfg.kwargs.get("BLOCK_SIZE"),
                "num_warps":  cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
            for key, cfg in cache.items()
        }
    except AttributeError:
        return {}


def export_autotune_results(output_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Export autotune results to JSON and return the results dict.

    Parameters
    ----------
    output_path:
        Path to write JSON.  Defaults to ``artifacts/autotune/rmsnorm_triton.json``.
    """
    results = get_autotune_best_configs()

    if output_path is None:
        output_path = Path("artifacts") / "autotune" / "rmsnorm_triton.json"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"operator": "rmsnorm", "implementation": "triton", "configs": results}, f, indent=2)

    return results
