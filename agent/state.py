
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
