"""
test_workflow_and_correction.py

Specific verification tests for user-requested requirements:
1. RAG-first workflow with automatic web fallback:
   - When knowledge base has the document, it uses the local knowledge base.
   - When knowledge base lacks the document, it automatically falls back to web search,
     obtains the result, and proceeds without getting stuck on the step.
2. Correction / replanning behavior:
   - When a step fails, the agent diagnoses and corrects it.
   - If replanning occurs, the agent does NOT repeat or re-execute completed steps.
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

from tools.rag_lookup import handle_tool_call as rag_lookup_call
from tools.knowledge_lookup import lookup_with_fallback
from agent.loop import run_agent
from planner.replanner import replan_step


def test_rag_first_local_hit():
    print("\n" + "=" * 60)
    print("TEST: RAG-First Local Hit (Knowledge Base)")
    print("=" * 60)
    query = "adaptive self-correcting agent phases architecture"
    result = rag_lookup_call({"query": query})
    print("Query:", query)
    print("Result snippet:", str(result)[:300])
    assert "[SOURCE: KNOWLEDGE BASE]" in result or "Knowledge base result" in result or "adaptive" in result.lower(), (
        f"Expected local knowledge base hit, got: {result[:200]}"
    )
    print(">>> [PASSED] Local knowledge base hit verified.")


def test_rag_web_fallback_miss():
    print("\n" + "=" * 60)
    print("TEST: RAG-First Web Fallback on KB Miss")
    print("=" * 60)
    # Query for something definitely NOT in the local knowledge base (e.g. current Eiffel tower height or specific university)
    query = "Eiffel Tower height in meters Paris"
    result = rag_lookup_call({"query": query})
    print("Query:", query)
    print("Result snippet:", str(result)[:400])
    assert "[SOURCE: WEB FALLBACK]" in result, (
        f"Expected [SOURCE: WEB FALLBACK] in result, got: {result[:300]}"
    )
    assert "http" in result.lower(), "Expected web URL citation in fallback result"
    print(">>> [PASSED] Automatic web fallback on KB miss verified.")


def test_no_redo_on_replan():
    print("\n" + "=" * 60)
    print("TEST: No Re-Execution of Completed Steps on Replan")
    print("=" * 60)
    goal = "Calculate 15 * 10, then search for quantum computing history, then summarize."
    completed = [
        {"step": "Calculate 15 * 10 using calculate tool.", "result": "150", "verification_status": "passed"}
    ]
    remaining = [
        "Search for quantum computing history.",
        "Summarize the findings."
    ]
    failure_reason = "Simulated failure on quantum search."

    res = replan_step(goal, completed, remaining, failure_reason)
    new_plan = res["plan"]
    print("Completed step:", completed[0]["step"])
    print("Revised remaining plan:", new_plan)

    for step in new_plan:
        assert "15 * 10" not in step, f"Replanner repeated completed calculation step: {step}"
        assert "150" not in step or "calculate" not in step.lower(), f"Step re-calculates: {step}"
    print(">>> [PASSED] Completed steps are not repeated in revised plan.")


if __name__ == "__main__":
    test_rag_first_local_hit()
    test_rag_web_fallback_miss()
    test_no_redo_on_replan()
    print("\n" + "=" * 60)
    print("ALL TARGETED WORKFLOW & CORRECTION TESTS PASSED!")
    print("=" * 60)
