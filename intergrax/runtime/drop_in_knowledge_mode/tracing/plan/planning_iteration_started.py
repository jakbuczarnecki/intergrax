# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerPlanningIterationStartedDiagV1(DiagnosticPayload):
    iterations_used: int
    same_plan_repeats: int
    has_replan_ctx: bool
    replan_reason: Optional[str]
    replan_attempt: Optional[int]

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.planning_iteration_started"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "same_plan_repeats": self.same_plan_repeats,
            "has_replan_ctx": self.has_replan_ctx,
            "replan_reason": self.replan_reason,
            "replan_attempt": self.replan_attempt,
        }
