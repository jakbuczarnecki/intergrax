# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerIterationCompletedContinueDiagV1(DiagnosticPayload):
    iterations_used: int
    last_plan_id: str
    replan_attempt: int

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.planner.iteration_completed_continue"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "last_plan_id": self.last_plan_id,
            "replan_attempt": self.replan_attempt,
        }
