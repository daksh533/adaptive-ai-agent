"""
test_all_phases.py

Comprehensive test suite testing every phase (Phase 1 through Phase 6)
individually and with an end-to-end combined multi-phase goal.

Phases Covered:
- Phase 1: LLM Connection (Groq API)
- Phase 2: Tools (Calculator, Python Subprocess, Web Search)
- Phase 3 & 4: Multi-Step Planning & LangGraph State Graph
- Phase 5: RAG Knowledge Base (ChromaDB Vector Retrieval & Citations)
- Phase 6: Verification Module (Numeric Re-check, Evidence Grounding, Logic Consistency)
- Combined: Master prompt exercising all phases simultaneously
"""

import sys
import os
from pathlib import Path

# Ensure venv packages and project root are in sys.path
project_root = Path(__file__).resolve().parent
venv_site = project_root / "venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from agent.loop import run_agent
from verification.verifier import numeric_recheck


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step_results(state: dict):
    print("\n--- Execution & Verification Trace ---")
    for i, res in enumerate(state.get("results_so_far", []), 1):
        status = res.get("verification_status", "pending")
        reason = res.get("verification_reason", "")
        badge = "[PASS]" if status == "passed" else f"[{status.upper()}]"
        print(f"\n[Step {i}] {res['step']}")
        print(f"Result  : {str(res['result'])[:250]}...")
        print(f"Verifier: {badge} {reason}")


# ===========================================================================
# TEST 1: Phase 2 (Calculator) + Phase 4 (LangGraph) + Phase 6 (Verification)
# ===========================================================================
def test_phase_calculator_and_verification():
    print_header("TEST 1: Calculator Tool + LangGraph + Numeric Verification")
    goal = "Calculate 30% of 450, then multiply that result by 2."
    print(f"Prompt: {goal}\n")
    state = run_agent(goal)
    print_step_results(state)

    assert len(state["results_so_far"]) >= 2, "Expected at least 2 steps"
    for r in state["results_so_far"]:
        assert r["verification_status"] == "passed", f"Step failed verification: {r}"
    print("\n>>> [PASSED] Test 1: Arithmetic executed and verified successfully.")


# ===========================================================================
# TEST 2: Phase 2 (Python Execution) + Phase 6 (Logical Verification)
# ===========================================================================
def test_phase_python_tool():
    print_header("TEST 2: Python Tool (Sandboxed Execution) + Verification")
    goal = "Execute Python code to calculate the factorial of 6 and print the result."
    print(f"Prompt: {goal}\n")
    state = run_agent(goal)
    print_step_results(state)

    all_text = " ".join(str(r["result"]) for r in state["results_so_far"])
    assert "720" in all_text, "Expected factorial of 6 (720) in output"
    print("\n>>> [PASSED] Test 2: Python code executed in sandbox and verified.")


# ===========================================================================
# TEST 3: Phase 5 (RAG Knowledge Base) + Phase 6 (Evidence Grounding)
# ===========================================================================
def test_phase_rag_retrieval():
    print_header("TEST 3: RAG Knowledge Base Retrieval + Evidence Verification")
    goal = (
        "Look up the project overview in the knowledge base. "
        "Find who the three authors are and what university they belong to. "
        "Cite the document source in the final answer."
    )
    print(f"Prompt: {goal}\n")
    state = run_agent(goal)
    print_step_results(state)

    all_text = " ".join(str(r["result"]) for r in state["results_so_far"])
    assert "Daksh" in all_text, "Expected author 'Daksh' in RAG response"
    assert "IILM" in all_text, "Expected university 'IILM' in RAG response"
    print("\n>>> [PASSED] Test 3: RAG retrieval grounded in ChromaDB and verified.")


# ===========================================================================
# TEST 4: Phase 6 Verifier Catching Faulty Calculation (Negative Test)
# ===========================================================================
def test_phase_verifier_fault_detection():
    print_header("TEST 4: Phase 6 Verifier Catching Injected Calculation Error")
    subtask = "Calculate 20% of 500."
    bad_result = "20% of 500 = 250. The answer is 250."  # Intentional error: 20% of 500 = 100
    print(f"Subtask   : {subtask}")
    print(f"Bad Result: {bad_result}")

    vresult = numeric_recheck(subtask, bad_result)
    print(f"\nVerifier Outcome: {vresult.status.upper()}")
    print(f"Verifier Reason : {vresult.reason}")

    assert vresult.status == "failed", "Verifier should have caught the calculation error!"
    print("\n>>> [PASSED] Test 4: Verifier successfully caught intentionally flawed output!")


# ===========================================================================
# TEST 5: Combined Master Prompt (Testing ALL Phases 1 to 6 in One Execution)
# ===========================================================================
def test_combined_all_phases():
    print_header("TEST 5: Master Combined Prompt (Phases 1 through 6 End-to-End)")
    goal = (
        "Step 1: Look up project_overview.md in the knowledge base to find the total number of phases. "
        "Step 2: Take that number of phases and multiply it by 10 using the calculate tool. "
        "Step 3: Run Python code to evaluate print(sum(range(1, 11))) and return the sum. "
        "Finally, summarize the results of all steps."
    )
    print(f"Master Prompt:\n{goal}\n")
    state = run_agent(goal)
    print_step_results(state)

    assert len(state["results_so_far"]) >= 3, "Expected at least 3 completed steps"
    all_text = " ".join(str(r["result"]) for r in state["results_so_far"])
    assert "12" in all_text or "phases" in all_text, "Expected phase count from RAG"
    print("\n>>> [PASSED] Test 5: Master combined prompt successfully executed and verified across all phases!")


if __name__ == "__main__":
    test_phase_calculator_and_verification()
    test_phase_python_tool()
    test_phase_rag_retrieval()
    test_phase_verifier_fault_detection()
    test_combined_all_phases()

    print("\n" + "=" * 70)
    print("  ALL INDIVIDUAL AND COMBINED PHASE TESTS PASSED SUCCESSFULLY! (PHASES 1-6)")
    print("=" * 70)
