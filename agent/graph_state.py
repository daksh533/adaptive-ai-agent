from typing import Any
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph graph.

    Fields
    ------
    goal : str
        The original user goal passed to run_agent().

    plan : list[str]
        Ordered list of subtask strings produced by the Planner node.

    current_step : int
        Zero-based index of the subtask currently being (or about to be) executed.

    results_so_far : list[dict[str, Any]]
        Accumulated results from completed subtasks.
        Each entry: {"step": <subtask str>, "result": <result str>}

    verification_status : str
        Status of the most recent verification check.
        Values: "pending" | "passed" | "failed"
        Set by verifier_node (Phase 6 onward).

    verification_reason : str
        Human-readable reason returned by the last verifier run.
        Empty string when status is "pending".

    error_diagnosis : str
        Error category assigned by the Diagnoser node.
        Empty string until Phase 7.

    correction_attempts : int
        How many times the Correction Engine has retried the current step.
        Reset to 0 on each new step; used from Phase 8 onward.

    is_done : bool
        Set to True by the Executor node once all plan steps are finished.
    """

    goal: str
    plan: list[str]
    current_step: int
    results_so_far: list[dict[str, Any]]
    verification_status: str    # "pending" | "passed" | "failed"
    verification_reason: str    # filled by verifier_node (Phase 6)
    error_diagnosis: str        # filled from Phase 7 onward
    correction_attempts: int    # filled from Phase 8 onward
    is_done: bool
