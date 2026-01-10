# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerStaticExecutionRequestedReplanDiagV1(DiagnosticPayload):
    replans_used: int
    replan_reason: Optional[str]
    last_plan_id: str

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.static_execution_requested_replan"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replans_used": self.replans_used,
            "replan_reason": self.replan_reason,
            "last_plan_id": self.last_plan_id,
        }
