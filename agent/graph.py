"""
agent/graph.py

Builds the LangGraph state graph for the adaptive agent.

Phase 4 graph (current):
    START → planner_node → executor_node ─┐
                                ↑          │ (more steps)
                                └──────────┘
                                           │ (done)
                                          END

Future phases will insert nodes between executor_node and the
"more steps / done" decision:
    executor_node → risk_assessor → verifier → diagnoser → correction → ...
"""

from langgraph.graph import StateGraph, END

from agent.graph_state import GraphState
from agent.planner import create_plan
from agent.executor import execute_step


# ---------------------------------------------------------------------------
# Node: Planner
# ---------------------------------------------------------------------------

def planner_node(state: GraphState) -> dict:
    """
    Ask the LLM to break the user's goal into an ordered list of subtasks.
    Initialises all state fields that downstream nodes depend on.
    """
    print("\n[Planner] Creating plan...")

    plan = create_plan(state["goal"])

    print("\n[Planner] Plan created:")
    for i, step in enumerate(plan, start=1):
        print(f"  {i}. {step}")

    return {
        "plan": plan,
        "current_step": 0,
        "results_so_far": [],
        "verification_status": "pending",
        "error_diagnosis": "",
        "correction_attempts": 0,
        "is_done": False,
    }


# ---------------------------------------------------------------------------
# Node: Executor
# ---------------------------------------------------------------------------

def executor_node(state: GraphState) -> dict:
    """
    Execute the current subtask using the available tools.
    Advances current_step and sets is_done when all steps are finished.
    """
    plan = state["plan"]
    step_idx = state["current_step"]
    subtask = plan[step_idx]

    print(f"\n[Executor] Step {step_idx + 1}/{len(plan)}: {subtask}")

    result = execute_step(subtask, state["results_so_far"])

    new_results = state["results_so_far"] + [result]
    next_step = step_idx + 1
    is_done = next_step >= len(plan)

    safe_result = str(result['result']).encode('ascii', errors='replace').decode('ascii')
    print(f"[Executor] Result: {safe_result}")

    return {
        "current_step": next_step,
        "results_so_far": new_results,
        "verification_status": "pending",   # reset for each step
        "correction_attempts": 0,           # reset for each step
        "is_done": is_done,
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_executor(state: GraphState) -> str:
    """
    Conditional edge: decide what happens after a step is executed.

    Phase 4:  done → END  |  more steps → executor (loop)
    Phase 6+: done → END  |  more steps → risk_assessor / verifier
    """
    if state["is_done"]:
        return "done"
    return "continue"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph():
    """
    Compile and return the LangGraph CompiledGraph.

    Call once at startup; re-use the returned object for every run_agent() call.
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)

    # Entry point
    graph.set_entry_point("planner")

    # planner always flows into the first executor call
    graph.add_edge("planner", "executor")

    # After executor: loop back or finish
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "continue": "executor",   # run the next subtask
            "done": END,              # all subtasks finished
        },
    )

    return graph.compile()
