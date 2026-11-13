"""
test_verification.py

Phase 6 Checkpoint Tests:

TEST 1: Verifier correctly passes a good result.
TEST 2: Verifier correctly catches a bad arithmetic result (numeric re-check).
TEST 3: Verifier correctly catches a logically off-topic result.
TEST 4: End-to-end agent run now includes verification status in every step result.

Run:
    python test_verification.py
"""

import sys
from pathlib import Path

# Ensure venv packages are reachable
project_root = Path(__file__).resolve().parent
venv_site = project_root / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from verification.verifier import (
    verify_step,
    numeric_recheck,
    logical_consistency_check,
)
from agent.loop import run_agent


# ---------------------------------------------------------------------------
# TEST 1: Good result — should PASS all checks
# ---------------------------------------------------------------------------
def test_good_result_passes():
    print("\n" + "=" * 60)
    print("TEST 1: Correct Result should PASS verification")
    print("=" * 60)

    subtask = "Calculate 25% of 800."
    result = "25% of 800 = 200. So the answer is 200."

    vresult = verify_step(subtask, result)

    print(f"\nFinal Verdict: {vresult.status.upper()}")
    assert vresult.status == "passed", (
        f"Expected 'passed' for a correct result. Got: {vresult.status} — {vresult.reason}"
    )
    print("[PASSED] TEST 1 - Correct result verified as PASSED.")


# ---------------------------------------------------------------------------
# TEST 2: Wrong arithmetic — Numeric Re-Check should FAIL
# ---------------------------------------------------------------------------
def test_wrong_arithmetic_fails():
    print("\n" + "=" * 60)
    print("TEST 2: Wrong Arithmetic should FAIL Numeric Re-Check")
    print("=" * 60)

    subtask = "Calculate 25% of 800."
    # Deliberately wrong: 25% of 800 is 200, NOT 350
    bad_result = "25% of 800 = 350. So the answer is 350."

    vresult = numeric_recheck(subtask, bad_result)

    print(f"\nNumeric Re-Check Verdict: {vresult.status.upper()}")
    print(f"Reason: {vresult.reason}")
    assert vresult.status == "failed", (
        f"Expected numeric_recheck to FAIL on wrong arithmetic. Got: {vresult.status}"
    )
    print("[PASSED] TEST 2 - Wrong arithmetic correctly caught as FAILED.")


# ---------------------------------------------------------------------------
# TEST 3: Off-topic result — Logical Consistency should FAIL
# ---------------------------------------------------------------------------
def test_offtopic_result_fails():
    print("\n" + "=" * 60)
    print("TEST 3: Off-topic Result should FAIL Logical Consistency Check")
    print("=" * 60)

    subtask = "Search the knowledge base for who the project authors are."
    # Completely irrelevant answer
    bad_result = "The capital of France is Paris and it is located in Western Europe."

    vresult = logical_consistency_check(subtask, bad_result)

    print(f"\nLogical Consistency Verdict: {vresult.status.upper()}")
    print(f"Reason: {vresult.reason}")
    assert vresult.status == "failed", (
        f"Expected logical_consistency_check to FAIL on off-topic result. Got: {vresult.status}"
    )
    print("[PASSED] TEST 3 - Off-topic result correctly caught as FAILED.")


# ---------------------------------------------------------------------------
# TEST 4: End-to-End Agent with Verification in GraphState
# ---------------------------------------------------------------------------
def test_agent_has_verification_in_state():
    print("\n" + "=" * 60)
    print("TEST 4: End-to-End Agent Run with Verification (Phase 6 Checkpoint)")
    print("=" * 60)

    goal = "Calculate 15% of 500, then add 25 to that result."
    state = run_agent(goal)

    print("\n" + "=" * 60)
    print("AGENT RESULTS WITH VERIFICATION")
    print("=" * 60)

    for i, res in enumerate(state["results_so_far"], 1):
        v_status = res.get("verification_status", "missing")
        v_reason = res.get("verification_reason", "none")
        icon = "OK" if v_status == "passed" else "FAIL"
        print(f"\n[Step {i}] {res['step']}")
        print(f"Result  : {res['result']}")
        print(f"Verified: [{icon}] {v_reason}")

    # Checkpoint: every result must include verification fields
    for res in state["results_so_far"]:
        assert "verification_status" in res, (
            "Missing 'verification_status' field in result dict!"
        )
        assert res["verification_status"] in ("passed", "failed"), (
            f"Unexpected verification_status: {res['verification_status']}"
        )

    print("\n[PASSED] Phase 6 Checkpoint: All step results contain verification status!")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_good_result_passes()
    test_wrong_arithmetic_fails()
    test_offtopic_result_fails()
    test_agent_has_verification_in_state()

    print("\n" + "=" * 60)
    print("ALL PHASE 6 TESTS PASSED!")
    print("=" * 60)

