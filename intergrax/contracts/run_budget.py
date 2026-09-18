# © Artur Czarnecki. All rights reserved.

"""Per-run execution budget limits (platform-neutral · TOOL-ENG / UE-8)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BudgetEnforcementMode(Enum):
    """Defines how runtime reacts when a budget is exceeded."""

    ABORT = "abort"
    HITL = "hitl"


@dataclass(frozen=True)
class RunBudget:
    """
    Hard limits applied per run_id.
    All limits are optional; None means 'no limit'.
    """

    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None

    max_llm_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None

    max_rag_invocations: Optional[int] = None
    max_websearch_invocations: Optional[int] = None

    max_wall_time_seconds: Optional[float] = None

    max_planner_iterations: Optional[int] = None
    max_replans: Optional[int] = None

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value is None:
                continue
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class BudgetPolicy:
    """Defines what runtime should do when a budget is exceeded."""

    enforcement_mode: BudgetEnforcementMode
    hitl_reason: Optional[str] = None
