"""
test_phases_7_8_9.py

Phase 7-9 Checkpoint Tests:

TEST 1: Diagnoser classifies calculation errors correctly
TEST 2: Diagnoser classifies tool errors correctly
TEST 3: Diagnoser classifies grounding errors correctly
TEST 4: Diagnoser classifies consistency errors correctly
TEST 5: Diagnoser classifies incomplete errors correctly
TEST 6: Corrector fixes calculation errors
TEST 7: Corrector fixes incomplete results
TEST 8: Replanner generates revised remaining plan
TEST 9: End-to-end agent run with diagnose/correct/replan nodes

Run:
    python test_phases_7_8_9.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
venv_site = project_root / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from verification.diagnoser import diagnose_step
from correction.corrector import correct_step
from planner.replanner import replan_step
from agent.loop import run_agent


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_diagnoser_calculation_error():
    print_header("TEST 1: Diagnoser — Calculation Error Classification")
    subtask = "Calculate 25% of 800."
    bad_result = "25% of 800 = 350."
    vreason = "Numeric re-check failed: computed 200 but result claims 350."
    prev = []

    d = diagnose_step(subtask, bad_result, vreason, prev)
    print(f"\nSubtask   : {subtask}")
    print(f"Result    : {bad_result}")
    print(f"Class     : {d['error_class']}")
    print(f"Reason    : {d['reason']}")
    print(f"Confidence: {d['confidence']}")

    assert d["error_class"] == "calculation_error", (
        f"Expected 'calculation_error', got '{d['error_class']}'"
    )
    print("[PASSED] TEST 1 — Calculation error classified correctly.")


def test_diagnoser_tool_error():
    print_header("TEST 2: Diagnoser — Tool Error Classification")
    subtask = "Search the web for latest AI news."
    bad_result = "Error: web_search tool timed out after 10 seconds."
    vreason = "Tool call failed with timeout."
    prev = []

    d = diagnose_step(subtask, bad_result, vreason, prev)
    print(f"\nSubtask   : {subtask}")
    print(f"Result    : {bad_result}")
    print(f"Class     : {d['error_class']}")
    print(f"Reason    : {d['reason']}")

    assert d["error_class"] == "tool_error", (
        f"Expected 'tool_error', got '{d['error_class']}'"
    )
    print("[PASSED] TEST 2 — Tool error classified correctly.")



def test_diagnoser_grounding_error():
    print_header("TEST 3: Diagnoser — Grounding Error Classification")
    subtask = "Find who the project authors are from the knowledge base."
    bad_result = "The authors are John Doe and Jane Smith."
    vreason = "Evidence grounding failed: no source cited from RAG knowledge base."
    prev = []

    d = diagnose_step(subtask, bad_result, vreason, prev)
    print(f"\nSubtask   : {subtask}")
    print(f"Result    : {bad_result}")
    print(f"Class     : {d['error_class']}")
    print(f"Reason    : {d['reason']}")

    assert d["error_class"] == "grounding_error", (
        f"Expected 'grounding_error', got '{d['error_class']}'"
    )
    print("[PASSED] TEST 3 — Grounding error classified correctly.")



def test_diagnoser_consistency_error():
    print_header("TEST 4: Diagnoser — Consistency Error Classification")
    subtask = "Summarize the project phases."
    bad_result = "There are 6 phases. Actually there are 12 phases total."
    vreason = "Logical consistency failed: contradictory phase counts (6 vs 12)."
    prev = []

    d = diagnose_step(subtask, bad_result, vreason, prev)
    print(f"\nSubtask   : {subtask}")
    print(f"Result    : {bad_result}")
    print(f"Class     : {d['error_class']}")
    print(f"Reason    : {d['reason']}")

    assert d["error_class"] == "consistency_error", (
        f"Expected 'consistency_error', got '{d['error_class']}'"
    )
    print("[PASSED] TEST 4 — Consistency error classified correctly.")


def test_diagnoser_incomplete_error():
    print_header("TEST 5: Diagnoser — Incomplete Error Classification")
    subtask = "List all three authors and their university."
    bad_result = "The authors are Daksh."
    vreason = "Result incomplete: missing two authors and university."
    prev = []

    d = diagnose_step(subtask, bad_result, vreason, prev)
    print(f"\nSubtask   : {subtask}")
    print(f"Result    : {bad_result}")
    print(f"Class     : {d['error_class']}")
    print(f"Reason    : {d['reason']}")

    assert d["error_class"] == "incomplete_error", (
        f"Expected 'incomplete_error', got '{d['error_class']}'"
    )
    print("[PASSED] TEST 5 — Incomplete error classified correctly.")


def test_corrector_calculation():
    print_header("TEST 6: Corrector — Calculation Error Strategy")
    subtask = "Calculate 10% of 250."
    bad_result = "10% of 250 = 500."
    error_class = "calculation_error"
    prev = []

    c = correct_step(subtask, bad_result, error_class, prev)
    print(f"\nSubtask     : {subtask}")
    print(f"Bad result  : {bad_result}")
    print(f"Error class : {error_class}")
    print(f"Corrected   : {c['corrected_result']}")
    print(f"Strategy    : {c['correction_strategy']}")

    assert c["corrected_result"] != bad_result, (
        "Corrected result should differ from the bad result."
    )
    assert "Correction failed" not in str(c["corrected_result"]), (
        f"Corrector returned an error: {c['corrected_result']}"
    )
    assert c["correction_strategy"] == "calculation_error"
    print("[PASSED] TEST 6 — Calculation correction produced a different result.")



def test_corrector_incomplete():
    print_header("TEST 7: Corrector — Incomplete Error Strategy")
    subtask = "List all three authors and their university."
    bad_result = "The authors are Daksh."
    error_class = "incomplete_error"
    prev = []

    c = correct_step(subtask, bad_result, error_class, prev)
    print(f"\nSubtask     : {subtask}")
    print(f"Bad result  : {bad_result}")
    print(f"Error class : {error_class}")
    print(f"Corrected   : {str(c['corrected_result'])[:200]}...")
    print(f"Strategy    : {c['correction_strategy']}")

    assert len(c["corrected_result"]) > len(bad_result), (
        "Corrected result should be more complete than the bad result."
    )
    assert "Correction failed" not in str(c["corrected_result"]), (
        f"Corrector returned an error: {c['corrected_result']}"
    )
    assert c["correction_strategy"] == "incomplete_error"
    print("[PASSED] TEST 7 — Incomplete correction produced a longer result.")


def test_replanner():
    print_header("TEST 8: Replanner — Dynamic Plan Revision")
    goal = "Calculate 25% of 800, then add 50, then search for authors."
    completed = [
        {"step": "Calculate 25% of 800.", "result": "200", "verification_status": "passed"},
    ]
    remaining = ["add 50 to that result.", "search for authors."]
    failure_reason = "Max correction attempts exceeded on step 2."

    result = replan_step(goal, completed, remaining, failure_reason)
    new_plan = result["plan"]
    print(f"\nGoal          : {goal}")
    print(f"Completed     : {len(completed)} step(s)")
    print(f"Remaining     : {len(remaining)} step(s)")
    print(f"New plan count: {len(new_plan)} step(s)")
    for i, s in enumerate(new_plan, 1):
        print(f"  {i}. {s}")

    assert isinstance(new_plan, list), "Replanner should return a plan list."
    assert len(new_plan) > 0, "Replanner should return at least one step."
    assert result["current_step"] == 0, "Replanner should restart at step 0."
    print("[PASSED] TEST 8 — Replanner generated a revised plan.")


def test_end_to_end_with_diagnose_correct_replan():
    print_header("TEST 9: End-to-End Agent Run (Phases 7-9 Graph Nodes)")
    goal = "Calculate 50% of 200, then add 100 to that result."
    print(f"\nGoal: {goal}\n")

    state = run_agent(goal)

    print("\n--- Execution & Verification Trace ---")
    for i, res in enumerate(state.get("results_so_far", []), 1):
        status = res.get("verification_status", "pending")
        reason = res.get("verification_reason", "")
        diag = res.get("error_diagnosis", "")
        badge = "[PASS]" if status == "passed" else f"[{status.upper()}]"
        print(f"\n[Step {i}] {res['step']}")
        print(f"Result  : {str(res['result'])[:200]}...")
        print(f"Verifier: {badge} {reason}")
        if diag:
            print(f"Diagnosis: {diag}")

    for res in state["results_so_far"]:
        assert "verification_status" in res, "Missing verification_status."
        assert "error_diagnosis" in res, "Missing error_diagnosis field."

    is_done = state.get("is_done", False)
    print(f"\nAgent finished: {is_done}")
    print("[PASSED] TEST 9 — End-to-end agent run completed with Phases 7-9 nodes.")



if __name__ == "__main__":
    test_diagnoser_calculation_error()
    test_diagnoser_tool_error()
    test_diagnoser_grounding_error()
    test_diagnoser_consistency_error()
    test_diagnoser_incomplete_error()
    test_corrector_calculation()
    test_corrector_incomplete()
    test_replanner()
    test_end_to_end_with_diagnose_correct_replan()

    print("\n" + "=" * 60)
    print("  ALL PHASES 7-9 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

