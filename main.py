"""
main.py

Interactive entry point for the Adaptive Self-Correcting AI Agent.

Powered by LangGraph and equipped with all tools:
- calculate     (deterministic arithmetic)
- web_search    (internet search via DuckDuckGo)
- run_python    (isolated Python subprocess)
- rag_lookup    (local knowledge-base vector retrieval)

Phases 1-9:
  Phases 1-4: LLM + tools + planner/executor via a LangGraph state graph
  Phase 5:    RAG knowledge-base lookup
  Phase 6:    Verification (numeric, evidence, logical consistency)
  Phase 7:    Error diagnoser (6-class taxonomy)
  Phase 8:    Targeted self-correction engine
  Phase 9:    Dynamic replanner with bounded retries
"""

import sys
from pathlib import Path

# 1. Make the project's venv packages importable without activating the venv.
project_root = Path(__file__).resolve().parent
venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages))

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(dotenv_path=project_root / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from agent.loop import run_agent


def print_header() -> None:
    print("=" * 60)
    print("  Adaptive Self-Correcting AI Agent  (Phases 1-9)")
    print("=" * 60)
    print("Enter a multi-step goal or question.")
    print("Type 'exit' or 'quit' to stop.\n")


def show_result(result: dict, index: int) -> None:
    """Pretty-print one entry from the execution history."""
    status = result.get("verification_status", "pending")
    reason = result.get("verification_reason", "")
    diagnosis = result.get("error_diagnosis", "")
    attempts = result.get("correction_attempts", 0)
    icon = "PASS" if status == "passed" else status.upper()

    print(f"\n[Step {index}] {result.get('step', '')}")
    print(f"Result    : {result.get('result', '')}")
    print(f"Verified  : [{icon}] {reason}")

    if diagnosis:
        print(f"Diagnosis : {diagnosis} (correction attempts: {attempts})")


def show_summary(state: dict) -> None:
    """Print the final status, counts, and the last good answer."""
    results = state.get("results_so_far", [])
    passed = [item for item in results if item.get("verification_status") == "passed"]
    agent_status = state.get("agent_status", "unknown")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Agent status      : {agent_status}")
    print(f"Plan steps        : {len(state.get('plan', []))}")
    print(f"Executed attempts : {len(results)}")
    print(f"Passed            : {len(passed)}")
    print(f"Replans used      : {state.get('replan_attempts', 0)}")

    if passed:
        print("\nFINAL ANSWER:")
        print(str(passed[-1].get("result", "")).strip())

    if agent_status != "completed":
        print("\n[!] The agent could not fully verify this goal. See the failed steps above.")


def handle_goal(goal: str) -> bool:
    """Run one goal and print the full execution + verification trace."""
    try:
        state = run_agent(goal)

        print("\n" + "=" * 60)
        print("FINAL RESULTS  (Verification & Self-Correction)")
        print("=" * 60)

        for index, result in enumerate(state.get("results_so_far", []), 1):
            show_result(result, index)

        show_summary(state)
        return state.get("agent_status") == "completed"

    except Exception as exc:
        print(f"\n[Error executing goal]: {exc}")
        return False


def main() -> None:
    # One-shot mode: python main.py "your goal here"
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:]).strip()
        if not goal:
            print("Usage: python main.py \"your goal here\"")
            sys.exit(2)
        sys.exit(0 if handle_goal(goal) else 1)

    print_header()

    while True:
        try:
            goal = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not goal:
            continue
        if goal.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        handle_goal(goal)


if __name__ == "__main__":
    main()
