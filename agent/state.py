"""
agent/state.py

Legacy dataclass from the early phases (before LangGraph was introduced).

The live agent uses :class:`agent.graph_state.GraphState` (a TypedDict) as the
LangGraph channel schema. This dataclass is kept only because the early phase
tests (``test_state.py``) use it.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    goal: str

    plan: list[str] = field(default_factory=list)

    current_step: int = 0

    results_so_far: list[dict[str, Any]] = field(
        default_factory=list
    )
