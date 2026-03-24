# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

GovernanceLayer = Literal["rag", "websearch", "tools"]


@dataclass(frozen=True)
class LegalToolPlanGovernanceClampDiagV1(DiagnosticPayload):
    """
    Emitted when organization governance degrades a Nexus layer on :class:`LegalToolPlan`.
    """

    layer: GovernanceLayer
    reason_code: str

    def redact(self) -> LegalToolPlanGovernanceClampDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.tool_plan_governance.clamp.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "reason_code": self.reason_code,
        }
