"""
GPU detection and device-property helpers.

These functions are the single source of truth for hardware information
throughout GPUForge.  They are called at benchmark start to populate the
gpu_devices table (Phase 7) and annotate every BenchmarkResult.
"""

from __future__ import annotations

from typing import Optional

import torch


def get_gpu_info(device_index: int = 0) -> Optional[dict]:
    """Return a structured dict of CUDA device properties.

    Returns ``None`` when no CUDA device is available so callers can
    distinguish "no GPU" from an error condition.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(device_index)
    cuda_version = torch.version.cuda or "unknown"

    return {
        "index": device_index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "multiprocessor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "warp_size": props.warp_size,
        # Shared memory limits help the autotuner prune invalid configs.
        "max_shared_memory_per_block_kb": props.max_shared_memory_per_block // 1024,
        "max_shared_memory_per_sm_kb": props.max_shared_memory_per_multiprocessor // 1024,
        "l2_cache_size_mb": round(props.l2_cache_size / 1024**2, 1),
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
    }


def get_gpu_name(device: torch.device) -> Optional[str]:
    """Return the GPU name string for the given device, or ``None`` on CPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else 0
    return torch.cuda.get_device_properties(idx).name


def list_all_gpus() -> list[dict]:
    """Return device properties for every visible CUDA device."""
    if not torch.cuda.is_available():
        return []
    return [get_gpu_info(i) for i in range(torch.cuda.device_count())]  # type: ignore[misc]


def require_cuda(reason: str = "This operation") -> None:
    """Raise a clear RuntimeError when no CUDA GPU is present.

    CPU-only environments can still run reference-implementation tests,
    API schema tests, and database migration tests.  Only operations that
    explicitly call require_cuda() are blocked.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"\n{'─' * 60}\n"
            f"{reason} requires a compatible NVIDIA GPU with CUDA support.\n"
            "No CUDA device was detected in this environment.\n\n"
            "CPU-only environments can still run:\n"
            "  • Unit tests for reference implementations\n"
            "  • API schema and database logic tests\n"
            "  • Frontend tests\n\n"
            "To run GPU benchmarks, use a machine with an NVIDIA GPU and\n"
            "the NVIDIA Container Toolkit (see docker/README.md).\n"
            f"{'─' * 60}"
        )
