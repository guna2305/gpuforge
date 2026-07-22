"""
Kernel latency timer with CUDA-event precision on GPU and perf_counter on CPU.

Why CUDA events?
────────────────
``time.perf_counter`` measures wall-clock time including Python overhead,
thread scheduling jitter, and any CPU work that overlaps the GPU kernel.
CUDA events are stamped by the GPU command processor at the exact moment
they are inserted into the stream, giving sub-microsecond kernel-only
timing that is unaffected by Python GIL release latency.

Measurement protocol
────────────────────
1. Warm up: run the function ``n_warmup`` times to fill GPU/CPU caches and
   JIT-compile any lazy operations.  The warm-up results are discarded.
2. Synchronise before timing starts to drain any queued work.
3. For each iteration: record a start event, call fn(), record an end event,
   synchronise, then call elapsed_time().
4. Return the list of per-iteration latencies in milliseconds.

The caller (BenchmarkHarness) aggregates the list into BenchmarkStats.
"""

from __future__ import annotations

import time
from typing import Callable

import torch


class KernelTimer:
    """Measures latency for a zero-argument callable in milliseconds.

    Parameters
    ----------
    device:
        The device the kernel runs on.  Determines whether CUDA events or
        perf_counter is used.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Use CUDA events only when the target device is a CUDA device AND
        # a CUDA GPU is actually present in this environment.
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()

    def measure_latencies_ms(
        self,
        fn: Callable[[], object],
        n_warmup: int = 20,
        n_iter: int = 100,
    ) -> list[float]:
        """Run *fn* and return per-iteration latencies in milliseconds.

        Parameters
        ----------
        fn:
            Zero-argument callable.  Its return value is ignored.
        n_warmup:
            Number of warm-up calls before timing begins.  These fill
            GPU L2 cache and JIT-compile any lazy kernels.
        n_iter:
            Number of timed iterations.  Must be >= 2 for stats to be valid.

        Returns
        -------
        list[float]
            Latencies in milliseconds, one per iteration.
        """
        if n_iter < 2:
            raise ValueError(f"n_iter must be >= 2, got {n_iter}")

        # ── warm-up ──────────────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(n_warmup):
                fn()
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        # ── timed iterations ─────────────────────────────────────────────────
        latencies: list[float] = []

        if self.use_cuda:
            with torch.no_grad():
                for _ in range(n_iter):
                    # Create fresh event objects each iteration so prior
                    # timings can't contaminate this measurement.
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)

                    # record() inserts a timestamp into the current CUDA stream.
                    start_evt.record(torch.cuda.current_stream(self.device))
                    fn()
                    end_evt.record(torch.cuda.current_stream(self.device))

                    # synchronize() blocks until end_evt is reached; only after
                    # this call is elapsed_time() accurate.
                    torch.cuda.synchronize(self.device)
                    latencies.append(start_evt.elapsed_time(end_evt))
        else:
            # CPU fallback: perf_counter has ~100 ns resolution on most OSes.
            with torch.no_grad():
                for _ in range(n_iter):
                    t0 = time.perf_counter()
                    fn()
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1_000.0)  # s → ms

        return latencies
