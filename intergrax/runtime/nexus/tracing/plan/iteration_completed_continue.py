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

    def redact(self) -> PlannerIterationCompletedContinueDiagV1:
        """
        This diagnostic payload contains only planner iteration metadata
        (counters and internal plan identifiers).
        It does not include user content or tool outputs and is considered PII-safe.
        """
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.planner.iteration_completed_continue"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "last_plan_id": self.last_plan_id,
            "replan_attempt": self.replan_attempt,
        }
