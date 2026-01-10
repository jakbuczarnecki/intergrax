# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerPlanSourceFailedDiagV1(DiagnosticPayload):
    plan_source_type: str
    error_type: str
    error_message: str

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.plan_source_failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_source_type": self.plan_source_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
