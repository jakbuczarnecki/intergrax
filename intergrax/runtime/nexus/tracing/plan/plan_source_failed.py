# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class PlannerPlanSourceFailedDiagV1(DiagnosticPayload):
    plan_source_type: str
    error_type: str
    error_message: str

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.planner.plan_source_failed"

    def redact(self) -> PlannerPlanSourceFailedDiagV1:
        """
        This diagnostic payload may contain user content inside error_message.
        In production, error_message must not be persisted in raw form.
        We preserve structural metadata but remove the free-form message.
        """
        return PlannerPlanSourceFailedDiagV1(
            plan_source_type=self.plan_source_type,
            error_type=self.error_type,
            error_message=DEFAULT_REDACTED_TEXT,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_source_type": self.plan_source_type,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
