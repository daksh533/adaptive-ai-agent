"""
agent/loop.py

Public entry point for running the agent.

Phase 6: graph now routes executor -> verifier -> (next executor | END)
The verifier inspects every step result for numeric accuracy,
evidence grounding, and logical consistency before committing it to state.
"""

from agent.graph import build_graph
from agent.graph_state import GraphState


# Build the graph once at import time so it is reused across calls.
_graph = build_graph()


def run_agent(goal: str) -> GraphState:
    """
    Run the complete multi-step agent for a given goal.

    Args:
        goal: The user's natural-language task description.

    Returns:
        The final GraphState dict after all subtasks are verified and complete.
        Key fields:
            state["plan"]                – ordered list of subtasks
            state["results_so_far"]     – list of {step, result, verification_status, verification_reason}
            state["verification_status"] – "passed" | "failed" of the last step
    """

    initial_state: GraphState = {
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results_so_far": [],
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
        "correction_attempts": 0,
        "is_done": False,
    }

    print(f"\n{'='*50}")
    print(f"Goal: {goal}")
    print(f"{'='*50}")

    final_state = _graph.invoke(initial_state)

    return final_state
