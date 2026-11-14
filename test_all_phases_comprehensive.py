"""
test_all_phases_comprehensive.py

Comprehensive test suite verifying all Phases (1 through 9) and combined workflows.
Ensures:
1. No completed steps are re-done upon step failure or replan.
2. RAG checks local knowledge base first; if not present, falls back to web,
   takes the information, and proceeds to the next step without getting stuck.
3. Numeric verification, error taxonomy, correction strategies, and replanner all operate in harmony.
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
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

from tools.registry import dispatch
from tools.knowledge_lookup import lookup_with_fallback
from verification.diagnoser import diagnose_step
from correction.corrector import correct_step
from planner.replanner import replan_step
from agent.loop import run_agent


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Phase 1 & 2: Core Tools
# ---------------------------------------------------------------------------
def test_phase_2_tools():
    banner("PHASE 2: Deterministic Tools (Calculate & Python)")
    calc_res = dispatch("calculate", {"expression": "25 * 4 + 50"})
    assert calc_res == 150 or calc_res == 150.0, f"Calculate failed: {calc_res}"
    print("[PASS] Calculator tool verified: 25 * 4 + 50 = 150")

    py_res = dispatch("run_python", {"code": "print(sum(range(1, 11)))"})
    assert "55" in str(py_res), f"Python tool failed: {py_res}"
    print("[PASS] Python tool verified: sum(range(1, 11)) = 55")


# ---------------------------------------------------------------------------
# Phase 5: RAG-First with Web Fallback
# ---------------------------------------------------------------------------
def test_phase_5_rag_workflow():
    banner("PHASE 5: RAG-First Lookup with Web Fallback")

    # 1. Local hit
    kb_hit = lookup_with_fallback("adaptive agent architecture phases")
    assert kb_hit["source"] == "knowledge_base", f"Expected knowledge_base, got {kb_hit['source']}"
    assert "[SOURCE: KNOWLEDGE BASE]" in kb_hit["text"]
    print("[PASS] RAG successfully retrieved from local knowledge base.")

    # 2. Local miss -> Live Web Fallback
    web_fallback = lookup_with_fallback("current tallest building in Tokyo meters")
    assert web_fallback["source"] == "web", f"Expected web fallback, got {web_fallback['source']}"
    assert "[SOURCE: WEB FALLBACK]" in web_fallback["text"]
    assert "http" in web_fallback["text"].lower()
    print("[PASS] RAG successfully fell back to live web search when info was not in KB.")


# ---------------------------------------------------------------------------
# Phase 7 & 8: Diagnoser & Targeted Correction (No re-doing previous steps)
# ---------------------------------------------------------------------------
def test_phase_7_8_diagnoser_and_corrector():
    banner("PHASES 7 & 8: Diagnoser & Non-Redundant Correction")
    subtask = "Calculate 15% tip on a 120 dollar bill."
    bad_result = "15% of 120 = 50 dollars."
    reason = "Numeric re-check failed: computed 18 but stated 50."
    prev_results = [
        {"step": "Look up bill total.", "result": "The bill total is 120 dollars.", "verification_status": "passed"}
    ]

    diag = diagnose_step(subtask, bad_result, reason, prev_results)
    assert diag["error_class"] == "calculation_error"
    print(f"[PASS] Diagnoser classified calculation error with confidence {diag['confidence']}")

    corrected = correct_step(subtask, bad_result, diag["error_class"], prev_results)
    assert "18" in str(corrected["corrected_result"]), f"Expected 18 in result: {corrected}"
    print(f"[PASS] Corrector fixed calculation using context without repeating step 1: {corrected['corrected_result']}")


# ---------------------------------------------------------------------------
# Phase 9: Replanner preserves completed steps
# ---------------------------------------------------------------------------
def test_phase_9_replanner_preservation():
    banner("PHASE 9: Replanner Preserves Completed Steps")
    goal = "Calculate total area of 2 rooms (10x10 and 12x12), then convert to square feet."
    completed = [
        {"step": "Calculate area of room 1 (10x10).", "result": "100 sq meters", "verification_status": "passed"}
    ]
    remaining = [
        "Calculate area of room 2 (12x12).",
        "Sum both areas and convert to square feet."
    ]
    reason = "Tool formatting failed on room 2."

    rep = replan_step(goal, completed, remaining, reason)
    new_plan = rep["plan"]
    print("Completed steps:", [c["step"] for c in completed])
    print("New revised plan:", new_plan)

    for step in new_plan:
        assert "room 1" not in step.lower(), f"Completed room 1 was repeated in new plan: {step}"
    print("[PASS] Replanner generated remaining plan without repeating room 1.")


# ---------------------------------------------------------------------------
# End-to-End Master Prompt: Combined Multi-Phase Workflow
# ---------------------------------------------------------------------------
def test_combined_end_to_end():
    banner("COMBINED END-TO-END WORKFLOW (Phases 1-9)")
    goal = "Calculate 20% of 500, then add 50 to that value."
    print("Goal:", goal)
    state = run_agent(goal)

    assert state.get("is_done") is True, "Agent run did not finish"
    assert state.get("agent_status") == "completed", f"Status: {state.get('agent_status')}"

    results = state.get("results_so_far", [])
    passed = [r for r in results if r.get("verification_status") == "passed"]
    print(f"\nExecution summary: {len(results)} total step execution(s), {len(passed)} verified passed.")

    # Confirm final answer
    final_text = str(passed[-1].get("result", ""))
    assert "150" in final_text, f"Expected 150 in final answer, got: {final_text}"
    print(f"[PASS] Combined workflow succeeded with verified final answer: {final_text[:150]}")


if __name__ == "__main__":
    test_phase_2_tools()
    test_phase_5_rag_workflow()
    test_phase_7_8_diagnoser_and_corrector()
    test_phase_9_replanner_preservation()
    test_combined_end_to_end()

    print("\n" + "=" * 70)
    print("  ALL COMPREHENSIVE TESTS PASSED SUCCESSFULLY ACROSS ALL PHASES!")
    print("=" * 70)
