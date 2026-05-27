# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LegalDecisionStepDiagV1(DiagnosticPayload):

    step_name: str

    # decision
    decision_status: Optional[str]
    decision_confidence: Optional[float]

    # risk
    legal_checks_count: int
    high_risk_count: int

    # policy
    policy_violations_count: int
    high_severity_violations: int

    # recommendations
    recommendations_count: int

    # enforcement
    decision_before: Optional[str]
    decision_after: Optional[str]
    enforcement_triggered: bool

    def redact(self) -> LegalDecisionStepDiagV1:
        # no sensitive text here → passthrough
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "decision_status": self.decision_status,
            "decision_confidence": self.decision_confidence,
            "legal_checks_count": self.legal_checks_count,
            "high_risk_count": self.high_risk_count,
            "policy_violations_count": self.policy_violations_count,
            "high_severity_violations": self.high_severity_violations,
            "recommendations_count": self.recommendations_count,
            "decision_before": self.decision_before,
            "decision_after": self.decision_after,
            "enforcement_triggered": self.enforcement_triggered,
        }