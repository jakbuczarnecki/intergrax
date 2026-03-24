# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LegalDecisionEnforcementStepDiagV1(DiagnosticPayload):
    step_name: str
    decision_status_before: str
    decision_status_after: str
    enforcement_modified: bool
    has_high_risk_checks: bool
    has_policy_violations: bool
    legal_checks_count: int
    policy_violations_count: int
    decision_confidence: Optional[float]

    def redact(self) -> LegalDecisionEnforcementStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.decision_enforcement.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "decision_status_before": self.decision_status_before,
            "decision_status_after": self.decision_status_after,
            "enforcement_modified": self.enforcement_modified,
            "has_high_risk_checks": self.has_high_risk_checks,
            "has_policy_violations": self.has_policy_violations,
            "legal_checks_count": self.legal_checks_count,
            "policy_violations_count": self.policy_violations_count,
            "decision_confidence": self.decision_confidence,
        }
