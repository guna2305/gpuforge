"""
GPUForge: GPU Kernel Autotuning and LLM Inference Performance Platform.

Development phases
──────────────────
Phase 1  Pure PyTorch RMSNorm reference, benchmark harness, correctness tests.
Phase 2  Triton RMSNorm kernel with autotuning.
Phase 3  CUDA C++ RMSNorm with custom PyTorch operator registration.
Phase 4  Fused residual+RMSNorm, SwiGLU, causal softmax.
Phase 5  Nsight Compute automation, bottleneck analyzer, roofline model.
Phase 6  Llama-style decoder, operator swap framework, e2e benchmarks.
Phase 7  FastAPI backend, PostgreSQL, React dashboard.
Phase 8  NVIDIA Triton Inference Server, DCGM, Prometheus, Grafana.
Phase 9  TensorRT comparison, regression detection, full documentation.
"""

__version__ = "0.1.0"
