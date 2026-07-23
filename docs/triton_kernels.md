# Triton Kernels — Phase 2

## Why Triton?

Triton is a Python-embedded DSL that compiles to PTX (NVIDIA GPU assembly).
It sits between PyTorch's eager mode and hand-written CUDA C++:

| | PyTorch eager | Triton | CUDA C++ |
|---|---|---|---|
| Development speed | Fast | Medium | Slow |
| Flexibility | Low | High | Highest |
| Debugging | Easy | Medium | Hard |
| Performance ceiling | ~80% of peak | ~95% of peak | ~100% of peak |
| Autotuning | None built-in | Built-in | Manual |

Triton lets us prototype fused kernels quickly and validate that the
algorithm is correct before committing to CUDA C++ in Phase 3.

---

## RMSNorm Kernel Design

### Algorithm: two-pass, tile-based

```
for each row (one Triton program per row):
    # Pass 1: compute mean of squares
    sum_sq = 0
    for tile in row:
        x_fp32 = load(tile).cast(fp32)
        sum_sq += sum(x_fp32 ** 2)
    inv_rms = rsqrt(sum_sq / N + eps)

    # Pass 2: normalise and store
    for tile in row:
        x_orig = load(tile)           # preserve dtype for cast-back
        w_orig = load(weight[tile])
        y = (x_orig.fp32 * inv_rms * w_orig.fp32).cast(x_orig.dtype)
        store(output[tile], y)
```

**Why two passes?**
The inverse RMS of a row cannot be computed until the full row has been
accumulated.  A single-pass approach would require streaming the accumulation
across a reduction tree that spans the full hidden_dim — possible but more
complex.  Phase 4 will explore single-pass fused kernels.

**Why loop over tiles?**
In Triton, `tl.arange(0, BLOCK_SIZE)` allocates `BLOCK_SIZE` registers per
thread.  If we required `BLOCK_SIZE >= hidden_dim` (single-block approach),
a hidden_dim of 8192 would demand 8192 registers per thread, causing register
spills to local memory (DRAM).  The tile loop keeps `BLOCK_SIZE` small while
still processing arbitrary hidden_dim values.

**Memory traffic**
- Pass 1: read x → 1 × hidden_dim × element_size bytes
- Pass 2: read x + read weight → 2 × hidden_dim × element_size bytes
- Write: output → 1 × hidden_dim × element_size bytes
- Total: 4 × hidden_dim × element_size per row
- (The reference model uses 3× because weight is often negligible for large batch×seq)

---

## Numerical Correctness

All intermediate computations are done in `float32` regardless of input dtype:

```python
x = tl.load(...).to(tl.float32)   # upcast before squaring
sum_sq += tl.sum(x * x, axis=0)   # accumulate in fp32
```

Without this upcast, `fp16` accumulation overflows when the partial sum of
squares exceeds ~65504 (~256 elements at unit variance).

The output is cast back to the input dtype before storing:

```python
tl.store(..., y_fp32.to(x_orig.dtype), ...)
```

This matches the reference implementation exactly and satisfies all
Phase 1 correctness tolerances.

---

## Autotuning

### What is tuned
- `BLOCK_SIZE` ∈ {128, 256, 512, 1024, 2048, 4096}
- `num_warps` ∈ {2, 4, 8, 16}
- `num_stages = 2` (fixed; controls Triton's software pipeliner)

### Key
The autotuner treats `N` (hidden_dim) as the key.  Each unique `N` value
gets its own benchmark search and cached result.

### How caching works
1. First call with `N=1024`: Triton benchmarks all 9 configs (with warmup),
   picks the fastest, and caches the result in `.triton/`.
2. Subsequent calls with `N=1024`: use the cached config directly.
3. Call with `N=4096`: triggers a new search for that N.

### Exporting results

```bash
python benchmarks/bench_rmsnorm.py --export-autotune
# Writes: artifacts/autotune/rmsnorm_triton.json
```

The JSON file records the winning `BLOCK_SIZE`, `num_warps`, and `num_stages`
for each `N` that has been benchmarked, enabling:
- Comparison across GPU architectures
- Input to the Phase 7 autotuning database
- Reproducibility of the chosen config

---

## Expected Performance (to be filled after GPU run)

| Shape | dtype | Reference (ms) | Triton (ms) | Speedup | Autotune config |
|---|---|---|---|---|---|
| 1×1×1024 | fp16 | *tbd* | *tbd* | *tbd* | *tbd* |
| 8×128×1024 | fp16 | *tbd* | *tbd* | *tbd* | *tbd* |

Run `python benchmarks/bench_rmsnorm.py --output artifacts/phase2_results.json`
on a GPU machine and paste results here.

---

## Known Limitations (Phase 2)

- No backward pass.  Gradient support is planned for Phase 3 with the CUDA kernel.
- No quantised dtype support (INT8, FP8).  Planned as an advanced phase.
- Triton autotuning cold-start (first call) takes ~5–30 seconds depending on
  the number of configs.  Subsequent calls use the cache.
- The two-pass approach reads global memory twice.  A fused single-pass kernel
  (Phase 4) will reduce this to one read + one write for fused operators.

---

## Phase 3 Preview

Phase 3 replaces the Triton kernel with a CUDA C++ kernel registered through
the official PyTorch dispatcher.  The Triton and CUDA results will be compared
side-by-side in the benchmark table.
