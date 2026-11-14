"""
agent/loop.py

Public entry point for running the agent.

The compiled LangGraph is built once at import time and reused across calls.
"""

from agent.graph import build_graph
from agent.graph_state import GraphState


# Build the graph once so it is reused across run_agent() calls.
_graph = build_graph()


def run_agent(goal: str) -> GraphState:
    """
    Run the complete multi-step agent for a given goal.

    Args:
        goal: The user's natural-language task description.

    Returns:
        The final GraphState after all subtasks are verified or the agent
        exhausts its correction/replan budget.

        Key fields:
            state["plan"]                – ordered list of subtasks
            state["results_so_far"]      – execution + verification history
            state["verification_status"] – status of the last verification
            state["agent_status"]        – "completed" | "failed"
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
        "replan_attempts": 0,
        "is_done": False,
        "agent_status": "running",
    }

    print(f"\n{'=' * 50}")
    print(f"Goal: {goal}")
    print(f"{'=' * 50}")

    return _graph.invoke(initial_state, config={"recursion_limit": 100})
