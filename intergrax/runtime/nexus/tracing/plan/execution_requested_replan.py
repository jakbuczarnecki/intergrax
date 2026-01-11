# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerExecutionRequestedReplanDiagV1(DiagnosticPayload):
    iterations_used: int
    replan_reason: Optional[str]
    last_plan_id: str

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.planner.execution_requested_replan"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "replan_reason": self.replan_reason,
            "last_plan_id": self.last_plan_id,
        }
