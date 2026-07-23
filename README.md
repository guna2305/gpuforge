# GPUForge

**GPU Kernel Autotuning and LLM Inference Performance Platform**

GPUForge is an open-source GPU performance engineering project that implements transformer operations from scratch — in pure PyTorch, Triton, and CUDA C++ — then automatically benchmarks, profiles, and tunes them across GPU architectures.

> **Phase 1 complete** — PyTorch RMSNorm reference, benchmark harness, property-based correctness tests. CUDA and Triton kernels follow in Phases 2–3.

---

## What It Does

Most LLM inference frameworks are generic. GPUForge answers the question generic frameworks can't: *how fast can this specific operation run on this specific GPU, and why isn't it already there?*

It does this by:

1. **Implementing** each transformer op three ways — PyTorch reference, Triton kernel, CUDA C++
2. **Validating** every implementation numerically before claiming any speedup
3. **Autotuning** block sizes, tile shapes, warp counts, and vector widths per GPU architecture
4. **Profiling** with Nsight Compute and classifying each kernel as memory-bound, compute-bound, or occupancy-limited
5. **Serving** the optimised model through NVIDIA Triton Inference Server
6. **Visualising** everything in a FastAPI + React dashboard with GPU telemetry via DCGM and Prometheus

---

## Architecture

```
Phase 7-9  |  React Dashboard  /  FastAPI  /  PostgreSQL
Phase 8    |  NVIDIA Triton Server  /  DCGM  /  Prometheus / Grafana
Phase 6    |  Llama-style Decoder  /  Operator Swap Framework
Phase 5    |  Nsight Compute  /  Bottleneck Analyser  /  Roofline
Phase 3-4  |  CUDA C++ Kernels  /  Custom PyTorch Operators
Phase 2    |  Triton Kernels  /  Autotuner
Phase 1  * |  PyTorch Reference  /  Benchmark Harness  /  Tests
```

---

## Operators (implemented in order)

| Operator | PyTorch | Triton | CUDA C++ |
|---|---|---|---|
| RMSNorm | Phase 1 ✓ | Phase 2 | Phase 3 |
| Fused Residual + RMSNorm | Phase 4 | Phase 4 | Phase 4 |
| SwiGLU gate | Phase 4 | Phase 4 | Phase 4 |
| Causal Softmax | Phase 4 | Phase 4 | Phase 4 |

---

## Quick Start

```bash
git clone https://github.com/guna2305/gpuforge.git
cd gpuforge

# CPU-only (no GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Run tests
pytest tests/ -v -m "not gpu"

# Run benchmark
python benchmarks/bench_rmsnorm.py --cpu-only
```

For GPU benchmarking, install the CUDA PyTorch build and run without `--cpu-only`. The project prints a clear error message if no NVIDIA GPU is detected.

---

## Correctness First

No implementation is called an optimisation until it passes all correctness tests. Every comparison is made in FP32 against the PyTorch reference.

| dtype | atol | rtol |
|---|---|---|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 5e-2 | 5e-2 |

Tests cover: random inputs, all-zeros, large/small magnitudes, power-of-two and odd dimensions, non-contiguous tensors, and 600+ randomised Hypothesis examples. NaN or Inf in the output is always a failure.

---

## Benchmark Methodology

- Seed 42 for all inputs
- 20 warm-up iterations before timing
- 100 timed iterations per configuration
- CUDA events for GPU timing (not wall-clock)
- Median, p95, throughput, effective bandwidth, and CV reported
- No fabricated numbers — all results come from actual runs stored in `artifacts/`

---

## Supported Hardware

NVIDIA Ampere (sm_80/86/87), Ada (sm_89), Hopper (sm_90). Volta and Turing best-effort. CPU-only environments can run all reference tests and API tests.

---

## Roadmap

| Phase | Goal |
|---|---|
| 2 | Triton RMSNorm + autotuning + first speedup table |
| 3 | CUDA C++ RMSNorm + PyTorch custom operator |
| 4 | Fused kernels: residual+RMSNorm, SwiGLU, causal softmax |
| 5 | Nsight Compute CLI, bottleneck classifier, roofline model |
| 6 | Llama-style decoder, end-to-end inference benchmarks |
| 7 | FastAPI + PostgreSQL + React dashboard |
| 8 | NVIDIA Triton Server, DCGM, Prometheus, Grafana |
| 9 | TensorRT, regression detection, full documentation |

---

## License

MIT
