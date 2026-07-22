"""
RMSNorm reference implementation — pure PyTorch.

Background
──────────
RMSNorm (Zhang & Sennrich, 2019) replaces the mean-and-variance
normalisation of LayerNorm with a simpler root-mean-square normalisation:

    RMS(x) = sqrt( (1/d) · Σᵢ xᵢ² + ε )
    yᵢ     = ( xᵢ / RMS(x) ) · wᵢ

Removing mean subtraction eliminates the bias parameter and reduces the
number of reduction passes, making RMSNorm cheaper than LayerNorm while
retaining comparable quality.  It is used in LLaMA, Mistral, Gemma, and
most modern open-source LLMs.

Design decisions
────────────────
1. Always upcast to float32 for the reduction.
   fp16 accumulation of squared values can overflow (max fp16 ≈ 65504,
   so x² overflows when |x| > 256) or lose precision near the fp16 range
   edges.  Production implementations (LLaMA, Mistral) do the same.

2. Use torch.rsqrt instead of 1 / torch.sqrt.
   rsqrt maps to a single hardware instruction on modern GPUs and is
   numerically equivalent for positive arguments.

3. Cast back to the original dtype only after multiplying by the weight.
   Casting earlier would quantise the normalised value before scaling,
   losing one ULP of precision.

4. eps is added inside rsqrt (not outside sqrt) — this is the standard
   convention; it prevents division by zero when all xᵢ = 0.

This file is the ground truth for all numerical comparisons in later phases.
Do not modify the algorithm without updating the tolerance table in
tolerances.py and re-running the full correctness test suite.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional RMSNorm: normalise the last dimension of *x*.

    Parameters
    ----------
    x:
        Input tensor of shape ``(..., hidden_dim)``.  Any number of leading
        batch/sequence dimensions is supported.  All real dtypes are accepted;
        the reduction is promoted to float32 regardless of the input dtype.
    weight:
        Learned scale (γ) of shape ``(hidden_dim,)``.  Must match
        ``x.shape[-1]`` and be on the same device as *x*.
    eps:
        Small positive constant added to the mean-of-squares before rsqrt to
        prevent division by zero.  Default 1e-6 matches LLaMA/Mistral.

    Returns
    -------
    torch.Tensor
        Normalised and scaled tensor with the same shape and dtype as *x*.

    Raises
    ------
    ValueError
        If ``weight.shape[0] != x.shape[-1]`` or ``eps <= 0``.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dimension, got shape {x.shape}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {weight.shape}")
    if weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"weight dimension ({weight.shape[0]}) must match "
            f"x last dimension ({x.shape[-1]})"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    original_dtype = x.dtype

    # ── upcast to fp32 for numerically stable mean-of-squares reduction ───────
    x_fp32 = x.float()
    weight_fp32 = weight.float()

    # mean(x²) over the last (hidden) dimension; keepdim for broadcasting
    mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)

    # rsqrt(mean_sq + eps) is the inverse RMS — a single fused instruction on GPU
    inv_rms = torch.rsqrt(mean_sq + eps)

    # Normalise, scale by learned weight, cast back to original dtype
    return (x_fp32 * inv_rms * weight_fp32).to(original_dtype)


class RMSNorm(nn.Module):
    """Drop-in RMSNorm module for use in transformer blocks.

    Weight is initialised to ones (identity scale at construction).

    Parameters
    ----------
    hidden_dim:
        Size of the last dimension of the input tensor.
    eps:
        Stability constant for the inverse-RMS computation.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.eps = eps
        # Initialised to ones so the module is a no-op at construction
        # (before any training or weight loading).
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_forward(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, eps={self.eps:.2e}"
