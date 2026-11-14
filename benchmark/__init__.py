"""
benchmark/__init__.py

Phase 11 — Empirical Benchmarking & Comparative Evaluation Module.

Evaluates and compares the three core agent configurations:
    1. Baseline (Plan-and-Solve with zero verification)
    2. Fixed Self-Correcting (Phase 9: Unconditional 3-way verification)
    3. Adaptive Self-Correcting (Phase 10: Dynamic risk-tiered verification)
"""

from benchmark.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask
from benchmark.metrics import BenchmarkMetrics, RunResult
from benchmark.evaluator import BenchmarkEvaluator, run_comparative_benchmark

__all__ = [
    "BENCHMARK_TASKS",
    "BenchmarkTask",
    "BenchmarkMetrics",
    "RunResult",
    "BenchmarkEvaluator",
    "run_comparative_benchmark",
]
