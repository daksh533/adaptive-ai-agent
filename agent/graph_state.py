from typing import Any, TypedDict


class GraphState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph graph.

    Fields
    ------
    goal : str
        The original user goal passed to run_agent().

    plan : list[str]
        Ordered list of subtask strings produced by the Planner node.
        Replaced by the Replanner when the plan is invalidated.

    current_step : int
        Zero-based index of the subtask currently being executed/verified.
        It only advances after the Verifier passes the step.

    results_so_far : list[dict[str, Any]]
        Accumulated execution history. Each entry contains:
            step, result, verification_status, verification_reason,
            error_diagnosis, correction_attempts.

    verification_status : str
        Status of the most recent verification check.
        Values: "pending" | "passed" | "failed".

    verification_reason : str
        Human-readable reason returned by the last verifier run.

    error_diagnosis : str
        Error category assigned by the Diagnoser for the current failure.

    correction_attempts : int
        Number of targeted correction attempts for the current step.
        Reset to 0 whenever a step passes.

    replan_attempts : int
        Number of full plan rewrites performed so far. Bounded by MAX_REPLANS.

    is_done : bool
        True once all plan steps have passed, or the agent gives up.

    agent_status : str
        High-level outcome: "running" | "completed" | "failed".
    """

    goal: str
    plan: list[str]
    current_step: int
    results_so_far: list[dict[str, Any]]
    verification_status: str
    verification_reason: str
    error_diagnosis: str
    correction_attempts: int
    replan_attempts: int
    is_done: bool
    agent_status: str
