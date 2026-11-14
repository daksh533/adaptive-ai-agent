"""
test_phase_11.py

Phase 11 — Empirical Benchmarking & Comparative Evaluation Tests

Covers:
  1. BenchmarkTask definitions & category coverage
  2. BenchmarkMetrics aggregation and efficiency calculations
  3. Graph verification_mode switching (baseline vs. fixed vs. adaptive)
  4. BenchmarkEvaluator execution and report generation (JSON + Markdown)
  5. Multi-configuration comparative metrics validation
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_task_by_id
from benchmark.metrics import BenchmarkMetrics, RunResult
from benchmark.evaluator import BenchmarkEvaluator


PASS = "✅ PASSED"
FAIL = "❌ FAILED"


def run_test(name: str, fn) -> bool:
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    print("─" * 60)
    try:
        fn()
        print(f"  {PASS}")
        return True
    except Exception as exc:
        print(f"  {FAIL}: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_task_definitions():
    """Verify benchmark tasks are properly registered across required categories."""
    assert len(BENCHMARK_TASKS) >= 4, f"Expected at least 4 tasks, got {len(BENCHMARK_TASKS)}"
    categories = {t.category for t in BENCHMARK_TASKS}
    assert any("arithmetic" in c.lower() or "math" in c.lower() for c in categories)
    assert any("grounding" in c.lower() or "knowledge" in c.lower() for c in categories)
    assert any("formatting" in c.lower() for c in categories)

    task = get_task_by_id("MATH_01")
    assert task is not None
    assert task.id == "MATH_01"

    # Test validator with mock state
    mock_passed = {
        "agent_status": "completed",
        "results_so_far": [{"result": "The final result is 120, then 170, and 340."}],
    }
    assert task.validate(mock_passed) is True

    mock_failed = {
        "agent_status": "failed",
        "results_so_far": [],
    }
    assert task.validate(mock_failed) is False
    print(f"  → Verified {len(BENCHMARK_TASKS)} benchmark tasks and validation logic.")


def test_metrics_aggregation():
    """Verify metrics calculation for accuracy, latency, recovery rate, and efficiency ratio."""
    metrics = BenchmarkMetrics(configuration="adaptive")

    r1 = RunResult(
        task_id="T1",
        task_category="Math",
        configuration="adaptive",
        success=True,
        latency_seconds=4.0,
        total_steps=2,
        completed_steps=2,
        verification_calls_count=2,
        correction_attempts_count=0,
        replan_count=0,
    )
    r2 = RunResult(
        task_id="T2",
        task_category="Research",
        configuration="adaptive",
        success=True,
        latency_seconds=6.0,
        total_steps=3,
        completed_steps=3,
        verification_calls_count=3,
        correction_attempts_count=1,
        replan_count=0,
        error_recovery_successes=1,
    )

    metrics.record_run(r1)
    metrics.record_run(r2)

    assert metrics.total_tasks == 2
    assert metrics.successful_tasks == 2
    assert metrics.success_rate_pct == 100.0
    assert metrics.total_latency_seconds == 10.0
    assert metrics.mean_latency_seconds == 5.0
    assert metrics.total_verification_calls == 5
    assert metrics.error_recovery_rate_pct == 100.0
    assert metrics.efficiency_ratio == 20.0  # 100 / 5.0

    d = metrics.to_dict()
    assert d["configuration"] == "adaptive"
    assert len(d["runs"]) == 2
    print(f"  → Verified metrics aggregation: success_rate={metrics.success_rate_pct}%, efficiency={metrics.efficiency_ratio}")


def test_baseline_configuration_bypass():
    """Verify that mode='baseline' bypasses verification and advances steps without checking."""
    from agent.loop import run_agent

    goal = "Calculate 10 + 20"
    state = run_agent(goal, mode="baseline")

    assert state.get("verification_mode") == "baseline"
    assert state.get("agent_status") == "completed"

    results = state.get("results_so_far", [])
    assert len(results) > 0
    for r in results:
        assert "bypassed" in r.get("verification_reason", "").lower()
    print(f"  → Baseline configuration verified: verification correctly bypassed.")


def test_adaptive_configuration_routing():
    """Verify that mode='adaptive' dynamically uses risk tiers."""
    from agent.loop import run_agent

    goal = "Format these names in alphabetical order: Zeno, Aristotle"
    state = run_agent(goal, mode="adaptive")

    assert state.get("verification_mode") == "adaptive"
    assert state.get("agent_status") == "completed"
    assert state.get("risk_level") in ("low", "medium", "high")
    print(f"  → Adaptive configuration verified with risk tier: {state.get('risk_level')}")


def test_evaluator_and_report_generation():
    """Run an automated benchmark test and check report generation."""
    task = BenchmarkTask(
        id="TEST_01",
        category="Test Category",
        goal="Calculate 50 + 50",
        expected_keywords=["100"],
    )

    evaluator = BenchmarkEvaluator(
        tasks=[task],
        configurations=["baseline", "adaptive"],
        output_dir="benchmark/test_results",
    )

    evaluator.run_all(delay_between_tasks=1.0)

    json_path = os.path.join(evaluator.output_dir, "benchmark_report.json")
    md_path = os.path.join(evaluator.output_dir, "benchmark_report.md")

    assert os.path.exists(json_path), f"Missing JSON report at {json_path}"
    assert os.path.exists(md_path), f"Missing Markdown report at {md_path}"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "Comparative Matrix" in md_text
    assert "Baseline" in md_text
    assert "Adaptive" in md_text
    print(f"  → Evaluator successfully generated reports at {evaluator.output_dir}")


def main():
    print("\n" + "=" * 60)
    print("  PHASE 11 — EMPIRICAL BENCHMARKING & EVALUATION TESTS  ")
    print("=" * 60)

    tests = [
        ("Task Suite Definitions & Coverage", test_task_definitions),
        ("Metrics & Data Model Aggregation",  test_metrics_aggregation),
        ("Baseline Configuration Bypass",     test_baseline_configuration_bypass),
        ("Adaptive Configuration Routing",    test_adaptive_configuration_routing),
        ("Evaluator & Report Generation",     test_evaluator_and_report_generation),
    ]

    passed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        time.sleep(1.5)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("  ALL TESTS PASSED — Phase 11 implementation complete!")
    else:
        print(f"  {len(tests) - passed} test(s) FAILED — check output above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
