"""
benchmark/metrics.py

Phase 11 — Evaluation Metrics & Data Models

Calculates and aggregates academic evaluation metrics across agent configurations:
    1. Task Success Rate (%)
    2. Error Recovery Rate (%)
    3. Latency (Wall-clock seconds)
    4. Verification Overhead (LLM calls vs. rule-based / bypassed checks)
    5. Efficiency Ratio (Success / Latency & Cost trade-off)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class RunResult:
    """Detailed record of a single benchmark task execution."""
    task_id: str
    task_category: str
    configuration: str  # "baseline" | "fixed" | "adaptive"
    success: bool
    latency_seconds: float
    total_steps: int
    completed_steps: int
    verification_calls_count: int = 0
    verification_checks_detail: dict[str, int] = field(default_factory=lambda: {
        "low_risk_runs": 0,
        "medium_risk_runs": 0,
        "high_risk_runs": 0,
        "baseline_skips": 0,
    })
    correction_attempts_count: int = 0
    replan_count: int = 0
    error_recovery_successes: int = 0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_category": self.task_category,
            "configuration": self.configuration,
            "success": self.success,
            "latency_seconds": round(self.latency_seconds, 2),
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "verification_calls_count": self.verification_calls_count,
            "verification_checks_detail": self.verification_checks_detail,
            "correction_attempts_count": self.correction_attempts_count,
            "replan_count": self.replan_count,
            "error_recovery_successes": self.error_recovery_successes,
            "error_message": self.error_message,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated evaluation metrics for a specific configuration."""
    configuration: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    success_rate_pct: float = 0.0
    total_latency_seconds: float = 0.0
    mean_latency_seconds: float = 0.0
    total_steps_executed: int = 0
    total_verification_calls: int = 0
    total_corrections_attempted: int = 0
    total_error_recoveries: int = 0
    error_recovery_rate_pct: float = 0.0
    efficiency_ratio: float = 0.0  # (success_rate / mean_latency) * 10
    runs: list[RunResult] = field(default_factory=list)

    def record_run(self, run: RunResult) -> None:
        """Add a single execution result and recompute aggregate statistics."""
        self.runs.append(run)
        self.total_tasks = len(self.runs)
        self.successful_tasks = sum(1 for r in self.runs if r.success)
        self.failed_tasks = self.total_tasks - self.successful_tasks
        self.success_rate_pct = round((self.successful_tasks / self.total_tasks) * 100.0, 1)

        self.total_latency_seconds = round(sum(r.latency_seconds for r in self.runs), 2)
        self.mean_latency_seconds = round(self.total_latency_seconds / self.total_tasks, 2)
        self.total_steps_executed = sum(r.completed_steps for r in self.runs)
        self.total_verification_calls = sum(r.verification_calls_count for r in self.runs)
        self.total_corrections_attempted = sum(r.correction_attempts_count for r in self.runs)
        self.total_error_recoveries = sum(r.error_recovery_successes for r in self.runs)

        if self.total_corrections_attempted > 0:
            self.error_recovery_rate_pct = round(
                (self.total_error_recoveries / self.total_corrections_attempted) * 100.0, 1
            )
        else:
            self.error_recovery_rate_pct = 100.0 if self.success_rate_pct == 100.0 else 0.0

        # Efficiency ratio: accuracy relative to latency
        if self.mean_latency_seconds > 0:
            self.efficiency_ratio = round((self.success_rate_pct / self.mean_latency_seconds), 2)
        else:
            self.efficiency_ratio = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate_pct": self.success_rate_pct,
            "total_latency_seconds": self.total_latency_seconds,
            "mean_latency_seconds": self.mean_latency_seconds,
            "total_steps_executed": self.total_steps_executed,
            "total_verification_calls": self.total_verification_calls,
            "total_corrections_attempted": self.total_corrections_attempted,
            "total_error_recoveries": self.total_error_recoveries,
            "error_recovery_rate_pct": self.error_recovery_rate_pct,
            "efficiency_ratio": self.efficiency_ratio,
            "runs": [r.to_dict() for r in self.runs],
        }
