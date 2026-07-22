"""
Shared pytest fixtures for the GPUForge test suite.

Device and dtype fixtures are parametrised so every test that depends on
them automatically runs on every supported configuration without duplicating
test code.

Marks
─────
@pytest.mark.gpu  — skip unless a CUDA device is present.
                    Applied automatically to any test using the ``cuda_device``
                    fixture, but can also be applied manually.
"""

from __future__ import annotations

import pytest
import torch


# ── Device fixtures ───────────────────────────────────────────────────────────

def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrised fixture: yields cpu and (if present) cuda."""
    return torch.device(request.param)


@pytest.fixture
def cpu_device() -> torch.device:
    """Always returns the CPU device."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """Returns the CUDA device; skips the test if none is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    return torch.device("cuda")


# ── Dtype fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def float_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Parametrised fixture: fp32, fp16, bf16."""
    return request.param  # type: ignore[return-value]


@pytest.fixture(params=[torch.float32, torch.float16])
def float_dtype_no_bf16(request: pytest.FixtureRequest) -> torch.dtype:
    """fp32 and fp16 only — for tests where bf16 needs separate handling."""
    return request.param  # type: ignore[return-value]


# ── Random seed fixture ───────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def fixed_seed() -> None:
    """Set a deterministic random seed for the duration of the test."""
    torch.manual_seed(42)


# ── pytest hooks ─────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring a CUDA-capable NVIDIA GPU",
    )
