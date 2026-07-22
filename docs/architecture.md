# GPUForge Architecture

## Overview

GPUForge is a layered GPU performance engineering platform.  Each layer
depends only on the layer below it, and every layer is independently testable.

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 7-9  React Dashboard  /  FastAPI  /  PostgreSQL          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 8    NVIDIA Triton Inference Server  /  DCGM  /  Grafana │
├─────────────────────────────────────────────────────────────────┤
│  Phase 6    Llama-style Decoder  /  Operator Swap Framework     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5    Nsight Compute  /  Bottleneck Analyzer  /  Roofline │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3-4  CUDA C++ Kernels  /  Custom PyTorch Operators       │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2    Triton Kernels  /  Triton Autotuner                 │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1 ★  PyTorch Reference  /  Benchmark Harness  /  Tests  │
└─────────────────────────────────────────────────────────────────┘
```

★ Current phase.

---

## Phase 1 Components

### `gpuforge/ops/reference/`

The **correctness ground truth**.  Pure PyTorch implementations that are
obviously correct by inspection.  These are never replaced by optimised
versions; instead, optimised implementations are compared against them.

```
rmsnorm.py        rmsnorm_forward()  and  RMSNorm(nn.Module)
tolerances.py     Per-dtype absolute and relative tolerances
```

**Key design choice — always upcast to fp32:**
The mean-of-squares reduction is always computed in float32, even when the
input is fp16 or bf16.  This prevents:

- `fp16` overflow when squaring values above ~256 (max fp16 ≈ 65504,
  max fp16² ≈ 4.3 × 10⁹ which overflows fp16's max of 65504).
- Catastrophic cancellation in the accumulation of many small squared values.

This matches the reference implementation in LLaMA, Mistral, and Gemma.

---

### `gpuforge/bench/`

The **benchmark harness** is independent of every operator.  It:

1. Warms up the GPU/CPU cache (`KernelTimer.measure_latencies_ms`).
2. Times the kernel using CUDA events on GPU, `perf_counter` on CPU.
3. Checks correctness against a provided FP32 reference tensor.
4. Derives effective memory bandwidth from the traffic model.
5. Packages everything into a `BenchmarkResult` Pydantic model.

```
timer.py      KernelTimer     — CUDA-event / CPU timing
harness.py    BenchmarkHarness — orchestration
models.py     BenchmarkStats, CorrectnessResult, BenchmarkResult, BenchmarkSuite
```

**Why CUDA events instead of Python timers?**
`time.perf_counter` captures wall-clock time including Python overhead,
kernel launch latency, and any CPU work that happens while the GPU is
running.  CUDA events are stamped by the GPU command processor at the
exact moment they are inserted into the stream, giving sub-microsecond,
kernel-only measurement that is independent of Python GIL release jitter.

---

### `gpuforge/utils/`

```
gpu_info.py   get_gpu_info(), get_gpu_name(), require_cuda()
```

`require_cuda()` is the single gating function for GPU-only paths.  It
prints a clear, actionable error message instead of a cryptic CUDA assertion.

---

### `benchmarks/`

Standalone scripts that exercise the harness.  They are kept **separate
from the `gpuforge` package** so they are never accidentally imported as
library code.  Each script accepts `--output` to save results as JSON.

---

### `tests/`

```
tests/
├── conftest.py                 Shared fixtures (device, dtype, seed)
├── ops/
│   └── test_rmsnorm_reference.py
└── bench/
    └── test_models.py
```

Tests are split into categories:
- **Deterministic** tests with fixed seeds.
- **Property-based** tests with Hypothesis (100+ random examples each).
- **Parametrised** over device (cpu, cuda) and dtype (fp32, fp16, bf16).

GPU tests use the `cuda_device` fixture which auto-skips when no GPU is
present, so the full suite can be run on CPU-only CI runners.

---

## Data Flow (Phase 1)

```
bench_rmsnorm.py
    │
    ├── create inputs (torch.randn, dtype, device)
    │
    ├── compute FP32 reference:  rmsnorm_forward(x_fp32, w_fp32)
    │
    ├── BenchmarkHarness.run()
    │       │
    │       ├── fn() once  →  correctness vs FP32 reference
    │       │
    │       ├── KernelTimer.measure_latencies_ms()
    │       │       ├── n_warmup calls (discarded)
    │       │       └── n_iter calls  →  list[float] (ms)
    │       │
    │       ├── compute_stats(latencies)  →  BenchmarkStats
    │       │
    │       └── BenchmarkResult (serialisable to JSON)
    │
    └── BenchmarkSuite  →  artifacts/bench.json
```

---

## Correctness Model

Every comparison is made in **float32**, regardless of the implementation's
working dtype.  This lets us compare fp16 outputs to fp32 references without
inflating errors due to dtype mismatch in the comparison itself.

| dtype    | atol   | rtol   | Rationale                                    |
|----------|--------|--------|----------------------------------------------|
| float32  | 1e-5   | 1e-5   | Nearly exact; absorbs instruction reordering |
| float16  | 1e-2   | 1e-2   | 10-bit mantissa; round-trip ~0.1% error      |
| bfloat16 | 5e-2   | 5e-2   | 7-bit mantissa; round-trip ~0.8% error       |

---

## Repository Layout

```
gpuforge/
├── ops/
│   ├── reference/    # Phase 1: PyTorch ground truth
│   ├── triton/       # Phase 2: Triton kernels
│   └── cuda/         # Phase 3: CUDA C++ kernels
├── bench/            # Timing, correctness, Pydantic models
├── profile/          # Phase 5: Nsight Compute, bottleneck analyser
├── models/           # Phase 6: Transformer decoder
└── utils/            # GPU info, logging

csrc/                 # Phase 3+: CUDA C++ source
  ops/
    rmsnorm/          # One subdirectory per operator
    fused_add_rmsnorm/

tests/                # pytest suite (CPU-safe by default)
benchmarks/           # Standalone benchmark scripts
artifacts/            # Generated results (gitignored)
docs/                 # Documentation
docker/               # Dockerfiles and Compose (Phase 8)
.github/workflows/    # CI/CD
```

---

## Phase Roadmap

| Phase | Description                                            | Status      |
|-------|--------------------------------------------------------|-------------|
| 1     | Reference implementation, harness, tests               | **Complete**|
| 2     | Triton RMSNorm + autotuning                            | Planned     |
| 3     | CUDA C++ RMSNorm + PyTorch custom operator             | Planned     |
| 4     | Fused residual+RMSNorm, SwiGLU, causal softmax         | Planned     |
| 5     | Nsight Compute, bottleneck analyser, roofline          | Planned     |
| 6     | Llama decoder, operator swap, e2e benchmarks           | Planned     |
| 7     | FastAPI, PostgreSQL, React dashboard                   | Planned     |
| 8     | Triton Serving, DCGM, Prometheus, Grafana              | Planned     |
| 9     | TensorRT, regression detection, full documentation     | Planned     |
