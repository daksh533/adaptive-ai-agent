"""
benchmark/benchmark_tasks.py

Phase 11 — Benchmark Task Suite

Standardized benchmark tasks across key academic and operational dimensions:
  1. Arithmetic / Compound Calculation (High Risk verification)
  2. Domain Knowledge Base / RAG Grounding (Medium Risk verification)
  3. Live Web Research & Computation (Medium/High Risk verification)
  4. Text Formatting & Summarisation (Low Risk verification)
  5. Multi-Step Reasoning with Potential Recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class BenchmarkTask:
    """Represents a single standardized evaluation task."""
    id: str
    category: str
    goal: str
    expected_keywords: list[str] = field(default_factory=list)
    validator: Callable[[dict], bool] | None = None
    description: str = ""

    def validate(self, final_state: dict) -> bool:
        """Evaluate whether the agent run successfully satisfied the task."""
        if self.validator:
            try:
                return bool(self.validator(final_state))
            except Exception:
                return False

        # Default keyword-based validation
        if not final_state or final_state.get("agent_status") != "completed":
            return False

        results_so_far = final_state.get("results_so_far", [])
        if not results_so_far:
            return False

        # Accumulate all step results into one string for inspection
        combined_text = " ".join(
            str(step.get("result", "")) for step in results_so_far
        ).lower()

        if not self.expected_keywords:
            return True

        return all(kw.lower() in combined_text for kw in self.expected_keywords)


# ---------------------------------------------------------------------------
# Task suite definitions
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="MATH_01",
        category="Arithmetic / High Risk",
        description="Multi-step chained arithmetic calculation.",
        goal="Calculate 15% of 800, then add 50, and multiply the result by 2.",
        expected_keywords=["120", "170", "340"],
    ),
    BenchmarkTask(
        id="RAG_01",
        category="Knowledge Base Grounding / Medium Risk",
        description="Retrieval of project error taxonomy classes from local ChromaDB.",
        goal="Look up the project error taxonomy from the knowledge base and list the 6 error classes.",
        expected_keywords=["calculation", "tool", "grounding", "consistency"],
    ),
    BenchmarkTask(
        id="FORMAT_01",
        category="Formatting / Low Risk",
        description="Pure text reorganization and formatting without arithmetic.",
        goal="Format the following project contributors into an alphabetical bulleted list: Tarkeshwar, Daksh, Abhinav.",
        expected_keywords=["abhinav", "daksh", "tarkeshwar"],
    ),
    BenchmarkTask(
        id="RESEARCH_01",
        category="Web Research & Math / Hybrid Risk",
        description="Web search combined with arithmetic ratio calculation.",
        goal="Find the population of Tokyo and New York City, calculate their sum, and state Tokyo's percentage.",
        expected_keywords=["tokyo", "population", "%"],
    ),
    BenchmarkTask(
        id="RECOVERY_01",
        category="Error Recovery & Invalidation",
        description="Multi-step dependent reasoning that exercises verification and correction.",
        goal="Calculate 25% of 600, then compute 10% of that result, and finally multiply by 4.",
        expected_keywords=["150", "15", "60"],
    ),
]


def get_task_by_id(task_id: str) -> BenchmarkTask | None:
    """Retrieve a benchmark task by its unique ID."""
    for task in BENCHMARK_TASKS:
        if task.id.lower() == task_id.lower():
            return task
    return None
