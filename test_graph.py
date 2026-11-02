"""
test_graph.py

Phase 4 checkpoint test:
  The same 3–4 step task from Phase 3 must produce a correct final
  answer, now running through the compiled LangGraph graph.

Run:
    python test_graph.py
"""

from agent.loop import run_agent


goal = (
    "Calculate 25% of 800, "
    "then calculate 15% of that result, "
    "then add 50 to the final result."
)

state = run_agent(goal)

print("\n" + "=" * 50)
print("FINAL RESULTS  (Phase 4 — LangGraph)")
print("=" * 50)

import sys
sys.stdout.reconfigure(encoding='utf-8')

for result in state["results_so_far"]:
    print(f"\nStep : {result['step']}")
    print(f"Result: {result['result']}")
