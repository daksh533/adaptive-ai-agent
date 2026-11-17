"""
agent/graph.py

LangGraph state graph for the adaptive, self-correcting agent.

Phase 6-9 control flow:

    START -> planner -> executor -> verifier
                                       |
                    +------------------+------------------+
                    | passed                              | failed
                    v                                     v
             more steps? -> executor              attempts left?
                    |                                /          \\
                    v                          diagnose        replan
                   END                              |              |
                                                    v              v
                                                corrector       executor
                                                    |              |
                                                    +-> verifier   +-> END (if empty)

Key invariant
-------------
``current_step`` only advances when a step *passes* verification. A failed
step keeps its index so the diagnoser/corrector operate on the correct subtask
and the corrected result is re-verified before the agent moves on.
"""

from langgraph.graph import END, StateGraph

from agent.executor import execute_step
from agent.graph_state import GraphState
from agent.planner import create_plan
from correction.corrector import correct_step
from planner.replanner import replan_step
from verification.diagnoser import diagnose_step
from verification.risk_classifier import classify_risk       # Phase 10
from verification.verifier import VerificationResult, verify_step, verify_step_adaptive  # Phase 10/11


MAX_CORRECTION_ATTEMPTS = 2
MAX_REPLANS = 2


def _new_result_entry(subtask: str) -> dict:
    """Create a result-history entry with every field populated."""
    return {
        "step": subtask,
        "result": "",
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
        "correction_attempts": 0,
    }


def _passed_results(results: list[dict]) -> list[dict]:
    """
    Return only results that are not known failures.

    After a replan, ``results_so_far`` still contains the failed attempts that
    triggered the rewrite. Those must not be fed back to the executor/verifier
    as trusted context. Entries without a status (early-phase history) are
    kept for backwards compatibility.
    """
    return [
        result
        for result in results
        if result.get("verification_status") != "failed"
    ]


# ---------------------------------------------------------------------------
# Node: Planner
# ---------------------------------------------------------------------------

def planner_node(state: GraphState) -> dict:
    """
    Ask the LLM to break the user's goal into an ordered list of subtasks.
    Initialises every state field that downstream nodes depend on.
    """
    print("\n[Planner] Creating plan...")

    plan = create_plan(state["goal"])

    print("\n[Planner] Plan created:")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. {step}")

    return {
        "plan": plan,
        "current_step": 0,
        "results_so_far": [],
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
        "correction_attempts": 0,
        "replan_attempts": 0,
        "is_done": False,
        "agent_status": "running",
        "verification_mode": state.get("verification_mode", "adaptive"),
    }


# ---------------------------------------------------------------------------
# Node: Executor
# ---------------------------------------------------------------------------

def executor_node(state: GraphState) -> dict:
    """
    Execute the current subtask using the available tools.
    Appends a new pending entry to ``results_so_far``.
    """
    plan = state["plan"]
    step_index = state["current_step"]

    if step_index >= len(plan):
        return {"is_done": True, "agent_status": "completed"}

    subtask = plan[step_index]
    print(f"\n[Executor] Step {step_index + 1}/{len(plan)}: {subtask}")

    step_output = execute_step(subtask, _passed_results(state["results_so_far"]))
    result_text = str(step_output.get("result", ""))

    safe_result = result_text.encode("ascii", errors="replace").decode("ascii")
    print(f"[Executor] Result: {safe_result[:300]}")

    entry = _new_result_entry(subtask)
    entry["result"] = result_text

    return {
        "results_so_far": list(state["results_so_far"]) + [entry],
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
    }


# ---------------------------------------------------------------------------
# Node: Verifier  (Phase 6)
# ---------------------------------------------------------------------------

def verifier_node(state: GraphState) -> dict:
    """
    Verify the current step result. On success, advance to the next step;
    on failure, keep ``current_step`` unchanged so correction targets the
    correct subtask.
    """
    plan = state["plan"]
    step_index = state["current_step"]

    if step_index >= len(plan):
        return {"is_done": True, "agent_status": "completed"}

    subtask = plan[step_index]
    latest_entry = state["results_so_far"][-1]
    result_text = str(latest_entry.get("result", ""))

    print(f"\n[Verifier] Verifying Step {step_index + 1}/{len(plan)}...")

    # Phase 11: Multi-configuration support (baseline / fixed / adaptive)
    mode = state.get("verification_mode", "adaptive")
    if mode == "baseline":
        print(f"[Verifier] Configuration: BASELINE — skipping verification for Step {step_index + 1}.")
        verdict = VerificationResult(
            status="passed",
            method="baseline_bypass",
            reason="Baseline configuration: verification bypassed.",
        )
        risk_level = "none"
    elif mode == "fixed":
        print(f"[Verifier] Configuration: FIXED (Phase 9) — running full 3-way verification...")
        verdict = verify_step(
            subtask=subtask,
            result=result_text,
            goal=state.get("goal", ""),
            previous_results=_passed_results(state.get("results_so_far", [])[:-1]),
        )
        risk_level = "high"
    else:
        # Phase 10: Adaptive risk-based verification
        risk_info = classify_risk(subtask, result_text)
        risk_level = risk_info["risk_level"]
        verdict = verify_step_adaptive(
            subtask=subtask,
            result=result_text,
            goal=state.get("goal", ""),
            previous_results=_passed_results(state.get("results_so_far", [])[:-1]),
            risk_level=risk_level,
        )

    updated_entry = dict(latest_entry)
    updated_entry.update(
        {
            "step": subtask,
            "result": result_text,
            "verification_status": verdict.status,
            "verification_reason": verdict.reason,
            "error_diagnosis": state.get("error_diagnosis", ""),
            "correction_attempts": state.get("correction_attempts", 0),
            "risk_level": risk_level,  # Phase 10: tier assigned
        }
    )

    updated_results = list(state["results_so_far"])
    updated_results[-1] = updated_entry

    status_label = "PASSED" if verdict.status == "passed" else "FAILED"
    print(f"[Verifier] Verdict: {status_label} - {verdict.reason}")

    if verdict.status == "passed":
        next_step = step_index + 1
        is_done = next_step >= len(plan)
        return {
            "results_so_far": updated_results,
            "current_step": next_step,
            "verification_status": "passed",
            "verification_reason": verdict.reason,
            "error_diagnosis": "",
            "correction_attempts": 0,  # reset per-step counter
            "is_done": is_done,
            "agent_status": "completed" if is_done else "running",
            "risk_level": risk_level,  # Phase 10
        }

    return {
        "results_so_far": updated_results,
        "current_step": step_index,  # do not skip the failed step
        "verification_status": "failed",
        "verification_reason": verdict.reason,
        "is_done": False,
        "agent_status": "running",
        "risk_level": risk_level,  # Phase 10
    }


# ---------------------------------------------------------------------------
# Node: Diagnoser  (Phase 7)
# ---------------------------------------------------------------------------

def diagnoser_node(state: GraphState) -> dict:
    """Classify the verification failure into one of six error categories."""
    plan = state["plan"]
    step_index = state["current_step"]
    subtask = plan[step_index]
    latest_entry = state["results_so_far"][-1]
    result_text = str(latest_entry.get("result", ""))

    print(f"\n[Diagnoser] Classifying failure for Step {step_index + 1}...")

    diagnosis = diagnose_step(
        subtask,
        result_text,
        state.get("verification_reason", ""),
        _passed_results(state.get("results_so_far", [])[:-1]),
    )
    print(f"[Diagnosis] {diagnosis['error_class']} - {diagnosis['reason']}")

    updated_entry = dict(latest_entry)
    updated_entry["error_diagnosis"] = diagnosis["error_class"]
    updated_results = list(state["results_so_far"])
    updated_results[-1] = updated_entry

    return {
        "error_diagnosis": diagnosis["error_class"],
        "results_so_far": updated_results,
    }


# ---------------------------------------------------------------------------
# Node: Corrector  (Phase 8)
# ---------------------------------------------------------------------------

def corrector_node(state: GraphState) -> dict:
    """Apply the targeted correction strategy and re-queue the step."""
    plan = state["plan"]
    step_index = state["current_step"]
    subtask = plan[step_index]
    latest_entry = state["results_so_far"][-1]
    result_text = str(latest_entry.get("result", ""))
    error_class = state.get("error_diagnosis", "unknown_error")
    attempts = state.get("correction_attempts", 0) + 1

    print(
        f"\n[Corrector] Attempt {attempts}/{MAX_CORRECTION_ATTEMPTS}: "
        f"correcting '{error_class}' for Step {step_index + 1}..."
    )

    correction = correct_step(
        subtask,
        result_text,
        error_class,
        _passed_results(state.get("results_so_far", [])[:-1]),
    )
    corrected_result = correction.get("corrected_result", result_text)

    updated_entry = dict(latest_entry)
    updated_entry.update(
        {
            "step": subtask,
            "result": corrected_result,
            "verification_status": "pending",
            "verification_reason": "",
            "error_diagnosis": error_class,
            "correction_attempts": attempts,
        }
    )
    updated_results = list(state["results_so_far"])
    updated_results[-1] = updated_entry

    print(f"[Corrector] Corrected: {str(corrected_result)[:200]}")

    return {
        "results_so_far": updated_results,
        "verification_status": "pending",
        "verification_reason": "",
        "correction_attempts": attempts,
        "current_step": step_index,
    }


# ---------------------------------------------------------------------------
# Node: Replanner  (Phase 9)
# ---------------------------------------------------------------------------

def replanner_node(state: GraphState) -> dict:
    """Rewrite the remaining plan after repeated failures without repeating completed steps."""
    plan = state["plan"]
    step_index = state["current_step"]
    goal = state.get("goal", "")

    completed = [
        result
        for result in state.get("results_so_far", [])
        if result.get("verification_status") == "passed"
    ]
    completed_subtasks = [r.get("step", "") for r in completed]
    remaining = plan[step_index:] if step_index < len(plan) else []
    failure_reason = state.get("verification_reason", "Max correction attempts exceeded")
    replan_attempts = state.get("replan_attempts", 0) + 1

    print(
        f"\n[Replanner] Rewriting remaining subtask(s) "
        f"after {len(completed)} completed step(s) (replan {replan_attempts}/{MAX_REPLANS})..."
    )

    new_state = replan_step(goal, completed, remaining, failure_reason)
    revised_remaining = new_state.get("plan", [])
    print(f"[Replanner] Revised remaining plan has {len(revised_remaining)} step(s).")

    full_plan = completed_subtasks + revised_remaining
    next_step_index = len(completed_subtasks)

    if revised_remaining:
        agent_status = "running"
        is_done = False
    else:
        has_unresolved = any(
            result.get("verification_status") == "failed"
            for result in state.get("results_so_far", [])
        )
        agent_status = "failed" if has_unresolved else "completed"
        is_done = True
        next_step_index = len(full_plan)

    return {
        "plan": full_plan,
        "current_step": next_step_index,
        "verification_status": "pending",
        "verification_reason": "",
        "error_diagnosis": "",
        "correction_attempts": 0,
        "replan_attempts": replan_attempts,
        "is_done": is_done,
        "agent_status": agent_status,
    }


# ---------------------------------------------------------------------------
# Node: Give up (bounded retries exhausted)
# ---------------------------------------------------------------------------

def give_up_node(state: GraphState) -> dict:
    """Stop the run cleanly when correction/replan limits are exhausted."""
    reason = state.get("verification_reason", "") or "Unknown failure"
    print(
        f"\n[Agent] Stopping after {state.get('correction_attempts', 0)} correction "
        f"attempt(s) and {state.get('replan_attempts', 0)} replan(s)."
    )
    return {
        "is_done": True,
        "agent_status": "failed",
        "verification_reason": (
            f"{reason} - stopped after maximum correction/replan attempts."
        ),
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_verifier(state: GraphState) -> str:
    if state.get("verification_status") == "passed":
        return "done" if state.get("is_done") else "continue"

    if state.get("correction_attempts", 0) < MAX_CORRECTION_ATTEMPTS:
        return "diagnose"

    if state.get("replan_attempts", 0) < MAX_REPLANS:
        print(
            f"\n[Router] Max correction attempts ({MAX_CORRECTION_ATTEMPTS}) "
            "reached - sending to Replanner."
        )
        return "replan"

    return "give_up"


def route_after_replanner(state: GraphState) -> str:
    if (
        state.get("is_done")
        or not state.get("plan")
        or state.get("current_step", 0) >= len(state.get("plan", []))
    ):
        return "done"
    return "executor"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph():
    """
    Compile and return the LangGraph CompiledGraph.

    Call once at startup and reuse the returned object for every run_agent()
    call.
    """
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("diagnoser", diagnoser_node)
    graph.add_node("corrector", corrector_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("give_up", give_up_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_edge("diagnoser", "corrector")
    graph.add_edge("corrector", "verifier")
    graph.add_edge("give_up", END)

    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "continue": "executor",
            "done": END,
            "diagnose": "diagnoser",
            "replan": "replanner",
            "give_up": "give_up",
        },
    )

    graph.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            "done": END,
        },
    )

    return graph.compile()
