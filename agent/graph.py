"""
agent/graph.py

Builds the LangGraph state graph for the adaptive agent.

Phase 6 graph (current):
    START -> planner_node -> executor_node -> verifier_node
                                                    |
                              (passed) <------------+-----------> (failed)
                                 |                                    |
                     (more steps: back to executor)        (next step, marked failed)
                                 |                                    |
                                END (is_done)                       END (is_done)

Future phases will extend the failed branch:
    verifier_node(failed) -> diagnoser -> corrector -> (retry executor)
"""

from langgraph.graph import StateGraph, END

from agent.graph_state import GraphState
from agent.planner import create_plan
from agent.executor import execute_step
from verification.verifier import verify_step


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
        "verification_reason": "",
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
    Appends the new result to results_so_far with status='pending'.
    Does not advance current_step — verifier_node handles advancement.
    """
    plan = state["plan"]
    step_idx = state["current_step"]
    subtask = plan[step_idx]

    print(f"\n[Executor] Step {step_idx + 1}/{len(plan)}: {subtask}")

    step_output = execute_step(subtask, state["results_so_far"])
    result_text = step_output.get("result", "")

    safe_result = str(result_text).encode("ascii", errors="replace").decode("ascii")
    print(f"[Executor] Result: {safe_result[:300]}")

    new_results = state["results_so_far"] + [
        {
            "step": subtask,
            "result": result_text,
            "verification_status": "pending",
            "verification_reason": "",
        }
    ]

    return {
        "results_so_far": new_results,
    }


# ---------------------------------------------------------------------------
# Node: Verifier  (Phase 6)
# ---------------------------------------------------------------------------

def verifier_node(state: GraphState) -> dict:
    """
    Phase 6 Verifier node.

    Runs three independent checks on the latest executor result:
        1. Numeric Re-Check
        2. Evidence Grounding Check
        3. Logical Consistency Check

    Updates the last entry in results_so_far with the verification verdict,
    advances current_step, and flags is_done when all steps have executed.
    """
    plan = state["plan"]
    step_idx = state["current_step"]
    subtask = plan[step_idx]

    latest_entry = state["results_so_far"][-1]
    result_text = str(latest_entry.get("result", ""))

    print(f"\n[Verifier] Verifying Step {step_idx + 1}/{len(plan)}...")

    vresult = verify_step(
        subtask=subtask,
        result=result_text,
        goal=state.get("goal", ""),
        previous_results=state.get("results_so_far", [])[:-1],
    )

    # Update the last entry in results_so_far with real verification outcome
    updated_results = list(state["results_so_far"])
    updated_results[-1] = {
        "step": subtask,
        "result": result_text,
        "verification_status": vresult.status,
        "verification_reason": vresult.reason,
    }

    next_step = step_idx + 1
    is_done = next_step >= len(plan)

    status_label = "PASSED" if vresult.status == "passed" else "FAILED"
    print(f"[Verifier] Verdict: {status_label} — {vresult.reason}")

    return {
        "results_so_far": updated_results,
        "current_step": next_step,
        "verification_status": vresult.status,
        "verification_reason": vresult.reason,
        "is_done": is_done,
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_verifier(state: GraphState) -> str:
    """
    Conditional edge after verifier_node.

    Phase 6 routing:
        is_done=True  -> END
        is_done=False -> executor  (next subtask)

    Phase 7+ will route failed steps to diagnoser instead of executor.
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
    graph.add_node("verifier", verifier_node)

    # Entry point
    graph.set_entry_point("planner")

    # planner -> executor (first subtask)
    graph.add_edge("planner", "executor")

    # executor -> verifier (always verify every result)
    graph.add_edge("executor", "verifier")

    # verifier -> next executor OR end
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "continue": "executor",   # run the next subtask
            "done": END,              # all subtasks finished
        },
    )

    return graph.compile()
