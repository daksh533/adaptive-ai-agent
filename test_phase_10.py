"""
test_phase_10.py

Phase 10 — Adaptive Risk-Based Verification Tests

Covers:
  1. RiskClassifier — LOW tier
  2. RiskClassifier — MEDIUM tier
  3. RiskClassifier — HIGH tier
  4. verify_step_adaptive() — LOW: no LLM calls (only numeric_recheck)
  5. verify_step_adaptive() — MEDIUM: numeric + grounding (1 LLM call)
  6. verify_step_adaptive() — HIGH: all three checks (same as Phase 6)
  7. End-to-end: risk levels assigned and correct checks run in a full run
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, time

# ── Make project root importable ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

PASS = "\u2705 PASSED"
FAIL = "\u274c FAILED"


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
        return False


# ════════════════════════════════════════════════════════════════════════════
# Tests 1–3: RiskClassifier tier detection
# ════════════════════════════════════════════════════════════════════════════

def test_risk_classifier_low():
    """Formatting / summarisation subtask → expects 'low' tier."""
    from verification.risk_classifier import classify_risk, RISK_TIERS

    subtask = "Format the previous answer as a numbered list"
    result  = "1. Item one\n2. Item two\n3. Item three"
    info = classify_risk(subtask, result)

    assert "risk_level" in info, "Missing risk_level key"
    assert info["risk_level"] in RISK_TIERS, f"Unknown tier: {info['risk_level']}"
    # Accept low or medium (some models may hedge), but NOT high for pure formatting
    assert info["risk_level"] in ("low", "medium"), (
        f"Expected low/medium for formatting task, got {info['risk_level']}"
    )
    print(f"  → Classified as: {info['risk_level'].upper()} — {info['reason']}")


def test_risk_classifier_medium():
    """Factual web-lookup subtask → expects 'medium' tier."""
    from verification.risk_classifier import classify_risk, RISK_TIERS

    subtask = "Search the web for the capital of France and report the result"
    result  = "The capital of France is Paris. Source: https://en.wikipedia.org/wiki/Paris"
    info = classify_risk(subtask, result)

    assert info["risk_level"] in RISK_TIERS, f"Unknown tier: {info['risk_level']}"
    assert info["risk_level"] in ("medium", "high"), (
        f"Expected medium/high for factual lookup, got {info['risk_level']}"
    )
    print(f"  → Classified as: {info['risk_level'].upper()} — {info['reason']}")


def test_risk_classifier_high():
    """Multi-step arithmetic subtask → expects 'high' tier."""
    from verification.risk_classifier import classify_risk, RISK_TIERS

    subtask = "Calculate 25% of 500, then add 50 to get the final answer"
    result  = "25% of 500 = 125. 125 + 50 = 175. The final answer is 175."
    info = classify_risk(subtask, result)

    assert info["risk_level"] in RISK_TIERS, f"Unknown tier: {info['risk_level']}"
    assert info["risk_level"] == "high", (
        f"Expected high for arithmetic task, got {info['risk_level']}"
    )
    print(f"  → Classified as: {info['risk_level'].upper()} — {info['reason']}")


# ════════════════════════════════════════════════════════════════════════════
# Tests 4–6: verify_step_adaptive() per tier
# ════════════════════════════════════════════════════════════════════════════

def test_adaptive_verify_low_risk():
    """LOW tier: only numeric_recheck runs (no LLM calls)."""
    from verification.verifier import verify_step_adaptive

    subtask = "Summarise the previous result in one sentence"
    result  = "The agent completed all tasks successfully."

    verdict = verify_step_adaptive(
        subtask=subtask,
        result=result,
        risk_level="low",
    )

    assert verdict.status in ("passed", "failed"), f"Unexpected status: {verdict.status}"
    # Method must reflect that only numeric check ran (not full 'all')
    assert verdict.method in ("numeric_recheck",), (
        f"Expected method 'numeric_recheck', got '{verdict.method}'"
    )
    print(f"  → Verdict: {verdict.status.upper()} via [{verdict.method}] — {verdict.reason}")


def test_adaptive_verify_medium_risk():
    """MEDIUM tier: numeric + grounding checks run (not logical consistency)."""
    from verification.verifier import verify_step_adaptive

    subtask = "Look up who wrote Romeo and Juliet and state the author"
    result  = "Romeo and Juliet was written by William Shakespeare."

    verdict = verify_step_adaptive(
        subtask=subtask,
        result=result,
        goal="Research classic literature",
        risk_level="medium",
    )

    assert verdict.status in ("passed", "failed"), f"Unexpected status: {verdict.status}"
    assert verdict.method in ("numeric+grounding",), (
        f"Expected method 'numeric+grounding', got '{verdict.method}'"
    )
    print(f"  → Verdict: {verdict.status.upper()} via [{verdict.method}] — {verdict.reason}")


def test_adaptive_verify_high_risk():
    """HIGH tier: all three checks run (same behaviour as Phase 6)."""
    from verification.verifier import verify_step_adaptive

    subtask = "Calculate 20% of 400"
    result  = "20% of 400 = 80"

    verdict = verify_step_adaptive(
        subtask=subtask,
        result=result,
        goal="Math calculation",
        risk_level="high",
    )

    assert verdict.status in ("passed", "failed"), f"Unexpected status: {verdict.status}"
    assert verdict.method == "all", (
        f"Expected method 'all' for HIGH risk, got '{verdict.method}'"
    )
    print(f"  → Verdict: {verdict.status.upper()} via [{verdict.method}] — {verdict.reason}")


# ════════════════════════════════════════════════════════════════════════════
# Test 7: End-to-end with risk tiers active
# ════════════════════════════════════════════════════════════════════════════

def test_end_to_end_with_risk_tiers():
    """
    Run a full agent task that involves arithmetic.
    Verify that:
      - risk_level is present in final state
      - agent_status is 'completed'
    """
    from agent.loop import run_agent

    state = run_agent("Calculate 25% of 500 and then add 50")

    assert "risk_level" in state, (
        "risk_level key missing from final GraphState — check loop.py initial state"
    )
    assert state["agent_status"] == "completed", (
        f"Expected agent_status='completed', got {state['agent_status']!r}"
    )
    print(f"  → Final risk_level : {state['risk_level']}")
    print(f"  → agent_status     : {state['agent_status']}")
    print(f"  → Steps completed  : {len(state['results_so_far'])}")


# ════════════════════════════════════════════════════════════════════════════
# Test runner
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  PHASE 10 — ADAPTIVE RISK-BASED VERIFICATION  ")
    print("=" * 60)

    tests = [
        ("Risk Classifier — LOW tier",         test_risk_classifier_low),
        ("Risk Classifier — MEDIUM tier",      test_risk_classifier_medium),
        ("Risk Classifier — HIGH tier",        test_risk_classifier_high),
        ("Adaptive Verify — LOW (no LLM)",     test_adaptive_verify_low_risk),
        ("Adaptive Verify — MEDIUM (1 LLM)",   test_adaptive_verify_medium_risk),
        ("Adaptive Verify — HIGH (full)",      test_adaptive_verify_high_risk),
        ("End-to-End with Risk Tiers",         test_end_to_end_with_risk_tiers),
    ]

    passed = 0
    for name, fn in tests:
        ok = run_test(name, fn)
        if ok:
            passed += 1
        # small pause to avoid rate-limiting back-to-back LLM calls
        time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("  ALL TESTS PASSED — Phase 10 implementation complete!")
    else:
        print(f"  {len(tests) - passed} test(s) FAILED — check output above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
