"""
run_benchmark.py

Phase 11 — Benchmark CLI Runner

Usage:
    python run_benchmark.py
    python run_benchmark.py --configs baseline adaptive
    python run_benchmark.py --task MATH_01
"""

import argparse
import sys
import os

# Set UTF-8 encoding on Windows
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.benchmark_tasks import BENCHMARK_TASKS, get_task_by_id
from benchmark.evaluator import BenchmarkEvaluator, CONFIGURATIONS


def main():
    parser = argparse.ArgumentParser(description="Adaptive Agent AI — Phase 11 Empirical Benchmarking")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["baseline", "fixed", "adaptive"],
        default=["baseline", "fixed", "adaptive"],
        help="Configurations to compare (default: all three)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task ID to run (e.g., MATH_01, RAG_01, FORMAT_01)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Throttle delay between tasks in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    tasks_to_run = BENCHMARK_TASKS
    if args.task:
        selected = get_task_by_id(args.task)
        if not selected:
            print(f"Error: Task ID '{args.task}' not found. Available tasks:")
            for t in BENCHMARK_TASKS:
                print(f"  - {t.id} ({t.category}): {t.goal[:50]}...")
            sys.exit(1)
        tasks_to_run = [selected]

    evaluator = BenchmarkEvaluator(
        tasks=tasks_to_run,
        configurations=args.configs,
    )
    evaluator.run_all(delay_between_tasks=args.delay)


if __name__ == "__main__":
    main()
