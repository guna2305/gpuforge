# GPUForge

**GPU Kernel Autotuning and LLM Inference Performance Platform**

GPUForge is a production-quality, open-source GPU performance engineering
platform built to demonstrate practical skills in CUDA C++, Triton, custom
PyTorch operators, kernel profiling, and LLM inference optimisation.

> **Phase 1 complete** — PyTorch RMSNorm reference, benchmark harness,
> and property-based correctness test suite.  CUDA and Triton kernels
> follow in Phases 2–3.

---

## Problem Statement

Deploying large language models at production scale requires operating at or
near hardware limits.  A transformer layer executes dozens of distinct kernel
types — normalisation, attention, feed-forward — each with different compute
and memory characteristics.  Generic frameworks leave significant performance
on the table because they cannot specialise kernels for every (GPU, batch,
sequence, dtype) combination.

GPUForge answers three questions:

1. How fast *can* a given operation run on a given GPU?
2. Why is it not already at that speed?
3. What configuration achieves the closest approach to the hardware ceiling?

---

## Why Kernel Fusion Matters

A fused kernel replaces multiple memory round-trips with a single pass.
For an RMSNorm followed by a residual add:

```
Unfused:  load x → compute RMS → store norm → load norm + residual → store
Fused:    load x → compute RMS + residual add → store
```

On an A100 PCIe (2 TB/s HBM bandwidth), a 1024-element fp16 RMSNorm with
B=1, T=1 transfers ~4 KB.  An unfused kernel pair costs two round-trips
through DRAM; a fused kernel costs one.  At scale — thousands of tokens,
hundreds of layers — this halves the normalisation latency.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 7-9  React Dashboard  /  FastAPI  /  PostgreSQL          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 8    NVIDIA Triton Server  /  DCGM  /  Prometheus        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 6    Llama Decoder  /  Operator Swap Framework           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 5    Nsight Compute  /  Bottleneck Analyser  /  Roofline │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3-4  CUDA C++ Kernels  /  Custom PyTorch Operators       │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2    Triton Kernels  /  Autotuner                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1 ★  PyTorch Reference  /  Benchmark Harness  /  Tests  │
└─────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for a detailed description
of every component.

---

## Supported Hardware

| GPU Family      | Compute Capability | Status          |
|-----------------|--------------------|-----------------|
| NVIDIA Ampere   | 8.0, 8.6, 8.7      | Target platform |
| NVIDIA Ada      | 8.9                | Target platform |
| NVIDIA Hopper   | 9.0                | Target platform |
| NVIDIA Volta    | 7.0, 7.2           | Best-effort     |
| NVIDIA Turing   | 7.5                | Best-effort     |
| CPU (no GPU)    | —                  | Tests only      |

**Driver requirement:** ≥ 525.x for CUDA 12.x.

The project fails with a clear message when no compatible NVIDIA GPU is
detected.  CPU-only environments can run reference-implementation tests,
API schema tests, and database tests.

---

## Installation

### Prerequisites

- Python 3.10–3.12
- PyTorch ≥ 2.3  ([pytorch.org/get-started](https://pytorch.org/get-started/locally/))
- NVIDIA GPU + CUDA ≥ 12.1 (for GPU benchmarks)
- CUDA Toolkit ≥ 12.1 (for CUDA C++ kernels, Phase 3+)

### CPU-only (development / CI)

```bash
git clone https://github.com/guna2305/gpuforge.git
cd gpuforge

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### GPU (recommended for benchmarking)

```bash
git clone https://github.com/guna2305/gpuforge.git
cd gpuforge

python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,gpu]"
```

---

## Quick Start

### Run the test suite

```bash
# CPU only — works without a GPU, takes ~30 s
pytest tests/ -v -m "not gpu"

# All tests including GPU (requires NVIDIA GPU)
pytest tests/ -v
```

### Run the reference benchmark

```bash
# Auto-detects GPU; falls back to CPU
python benchmarks/bench_rmsnorm.py

# Save results to JSON
python benchmarks/bench_rmsnorm.py --output artifacts/phase1_results.json

# CPU-only mode
python benchmarks/bench_rmsnorm.py --cpu-only
```

Example console output (actual numbers vary by hardware):

```
========================================================================
  GPUForge Phase 1 — RMSNorm Reference Benchmark
========================================================================
  GPU  : NVIDIA GeForce RTX 4090
  ...
------------------------------------------------------------------------
  Shape               dtype     Median       p95   BW (GB/s)    CV    OK?
------------------------------------------------------------------------
  1×1×256             float32    0.012ms   0.013ms        N/A   4.1%    ✓
  1×1×4096            float32    0.018ms   0.019ms        N/A   2.8%    ✓
  ...
------------------------------------------------------------------------
  All correctness checks passed.
```

> **Note:** The table above is an illustrative template.  All numbers in
> stored `artifacts/` files come from actual measured runs and are never
> fabricated.

---

## Correctness Methodology

Correctness is a harder requirement than performance in GPUForge.

For every operator and every implementation:

- The FP32 PyTorch reference is the ground truth.
- All comparisons are done in FP32 regardless of working dtype.
- Tolerances are defined per dtype based on mantissa precision:

| dtype    | atol   | rtol   |
|----------|--------|--------|
| float32  | 1e-5   | 1e-5   |
| float16  | 1e-2   | 1e-2   |
| bfloat16 | 5e-2   | 5e-2   |

- Tests cover: random values, all-zeros, large magnitudes (~10⁴),
  small magnitudes (~10⁻⁶), power-of-two dims, odd dims, non-contiguous
  tensors, invalid inputs.
- **100 randomised property-based test cases** per property, via Hypothesis.
- NaN and Inf are treated as failures, not warnings.
- `CorrectnessResult` records `max_abs_error`, `mean_abs_error`, and
  `max_rel_error` even when a test fails, so failures are diagnosable.

---

## Benchmark Methodology

- Fixed random seed (42) for all inputs.
- **20 warm-up iterations** before timing starts.
- **100 timed iterations** per configuration.
- CUDA events for GPU timing; `time.perf_counter` on CPU.
- Synchronisation at measurement boundaries.
- Statistics reported: median, mean, std, min, max, p95, CV.
- Effective memory bandwidth derived from a traffic model (read x + weight,
  write output); documented as an estimate, not a hardware counter.
- Speedup tables added in Phase 2 when the Triton kernel is available.

---

## Real Benchmark Results

*Will be populated after running on a physical NVIDIA GPU.*

Run the benchmark yourself and check your results into `artifacts/`:

```bash
python benchmarks/bench_rmsnorm.py --n-iter 200 \
    --output artifacts/$(date +%Y%m%d)_rmsnorm_reference.json
```

---

## Limitations (Phase 1)

- Only the PyTorch reference is implemented.  No CUDA or Triton kernels yet.
- No autotuner.  No Nsight Compute integration.
- No transformer decoder.
- No FastAPI backend or dashboard.
- Benchmark results are not persisted to a database.
- The project requires an NVIDIA GPU for GPU benchmarks.  No AMD/Intel support.
- TensorRT and TensorRT-LLM are out of scope for Phase 1.
- INT8 and FP8 are planned for a later advanced phase.

---

## Reproduction Commands

```bash
# 1. Clone
git clone https://github.com/guna2305/gpuforge.git && cd gpuforge

# 2. Install (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# 3. Test
pytest tests/ -v -m "not gpu"

# 4. Benchmark
python benchmarks/bench_rmsnorm.py --cpu-only
```

---

## What I Learned

### GPU Execution Hierarchy
A CUDA kernel launches a **grid** of **blocks**, each of which contains
**warps** of 32 threads.  All threads in a warp execute the same instruction
simultaneously (SIMT).  Divergent branches within a warp serialise execution.

### Memory Coalescing
Global memory accesses are most efficient when all threads in a warp access
a contiguous, aligned 128-byte segment.  Strided or scattered accesses
fragment transactions into multiple cache lines, reducing effective bandwidth.

### Shared Memory
On-chip SRAM (~100 KB per SM on A100) is ~100× faster than HBM.  Tiling
algorithms load a tile from global memory into shared memory once, then
re-use it many times from the fast on-chip buffer.

### Warp-Level Reductions
`__shfl_down_sync` and `__shfl_xor_sync` exchange values between lanes
within a warp without touching shared memory.  A tree-reduction using
shuffles performs a warp-wide sum in 5 steps with no synchronisation cost.

### Occupancy
Occupancy is the ratio of active warps to the maximum warps an SM can hold.
It is limited by registers per thread, shared memory per block, and the
block size.  Higher occupancy helps hide memory latency but doesn't always
improve throughput (the roofline model predicts the ceiling).

### Register Pressure
Each SM has 65536 registers shared among all resident threads.  Using too
many registers per thread reduces occupancy.  The compiler reports spills
to local memory (which goes to DRAM) when a kernel exceeds the register file.

### Kernel Fusion
Fusing adjacent kernels eliminates intermediate DRAM round-trips.  The
RMSNorm + residual-add fusion reduces the normalisation pass from two
DRAM reads and two writes to one read and one write.

### Launch Overhead
Every CUDA kernel launch incurs ~5–20 µs of CPU-side overhead.  For tiny
inputs (H=64, B=1), this overhead can dominate the kernel time.  Fusing
reduces the number of launches.

### Numerical Stability
fp16 accumulation of squared values overflows when |x| > ~256.  Always
upcasting the RMS reduction to fp32 costs one type-conversion instruction
per thread and prevents overflow entirely.  The same pattern applies to
attention score accumulation and softmax.

### Roofline Analysis
The roofline model plots achieved FLOP/s versus arithmetic intensity
(FLOPs / byte).  Operations below the ridge point are memory-bound;
above it they are compute-bound.  RMSNorm has very low arithmetic
intensity and is always memory-bound; the goal is to maximise effective
bandwidth, not raw throughput.

### Performance Measurement Pitfalls
- Timing without warm-up includes kernel compilation and cache cold-start.
- A single timing sample is dominated by noise; always report the median
  of at least 100 samples.
- `time.perf_counter` on the CPU measures wall time, not GPU kernel time.
- CUDA events measure only the GPU kernel; they exclude Python overhead
  and launch latency, which is what you want for micro-benchmarks.

---

## Roadmap

| Phase | Target                                              |
|-------|-----------------------------------------------------|
| 2     | Triton RMSNorm, autotuning, first speedup table     |
| 3     | CUDA C++ RMSNorm, PyTorch custom operator           |
| 4     | Fused residual+RMSNorm, SwiGLU, causal softmax      |
| 5     | Nsight Compute CLI, bottleneck classifier, roofline |
| 6     | Llama-style decoder, end-to-end inference benchmarks|
| 7     | FastAPI backend, PostgreSQL, React dashboard         |
| 8     | NVIDIA Triton Server, DCGM, Prometheus, Grafana     |
| 9     | TensorRT, regression detection, full documentation  |

---

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) (coming in Phase 9).

Key rules:
- Every new implementation must pass all existing correctness tests before
  any performance discussion.
- Never commit fabricated benchmark numbers.
- Never silence failing tests.

---

## License

MIT — see [LICENSE](LICENSE).
