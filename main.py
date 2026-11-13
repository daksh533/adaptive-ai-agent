"""
main.py

Main interactive entry point for the Adaptive Self-Correcting AI Agent.
Powered by LangGraph and equipped with all tools:
- calculate (deterministic arithmetic)
- web_search (internet search via DuckDuckGo)
- run_python (sandboxed Python code execution)
- rag_lookup (local knowledge base vector retrieval)

Phase 6: Every step result is now inspected by the Verifier before
being committed to state. Verification status is shown in the output.
"""

import sys
import os
from pathlib import Path

# 1. Automatically link venv packages so 'python main.py' works without activation
project_root = Path(__file__).resolve().parent
venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages))

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent.loop import run_agent


def main():
    print("=" * 60)
    print("  Adaptive Self-Correcting AI Agent  (Phase 6 — Verification)")
    print("=" * 60)
    print("Enter a multi-step goal or question.")
    print("Type 'exit' or 'quit' to stop.\n")

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

        try:
            state = run_agent(goal)

            print("\n" + "=" * 60)
            print("FINAL RESULTS  (Phase 6 — With Verification)")
            print("=" * 60)
            for i, res in enumerate(state.get("results_so_far", []), 1):
                v_status = res.get("verification_status", "?")
                v_reason = res.get("verification_reason", "")
                icon = "OK" if v_status == "passed" else "FAIL"
                print(f"\n[Step {i}] {res['step']}")
                print(f"Result  : {res['result']}")
                print(f"Verified: [{icon}] {v_reason}")
        except Exception as e:
            print(f"\n[Error executing goal]: {e}")


if __name__ == "__main__":
    main()
