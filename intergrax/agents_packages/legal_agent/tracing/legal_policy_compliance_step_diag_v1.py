# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

LegalPolicyComplianceOutcome = Literal["completed"]


@dataclass(frozen=True)
class LegalPolicyComplianceStepDiagV1(DiagnosticPayload):
    step_name: str
    outcome: LegalPolicyComplianceOutcome
    clauses_count: int
    violations_count: int
    high_severity_violations: int

    def redact(self) -> LegalPolicyComplianceStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.policy_compliance.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "outcome": self.outcome,
            "clauses_count": self.clauses_count,
            "violations_count": self.violations_count,
            "high_severity_violations": self.high_severity_violations,
        }
