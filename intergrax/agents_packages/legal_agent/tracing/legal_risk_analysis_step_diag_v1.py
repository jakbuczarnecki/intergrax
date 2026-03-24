# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LegalRiskAnalysisStepDiagV1(DiagnosticPayload):
    step_name: str
    clauses_input_count: int
    legal_checks_added_count: int
    sensitive_flags_added_count: int
    high_risk_checks_added_count: int

    def redact(self) -> LegalRiskAnalysisStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.risk_analysis.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "clauses_input_count": self.clauses_input_count,
            "legal_checks_added_count": self.legal_checks_added_count,
            "sensitive_flags_added_count": self.sensitive_flags_added_count,
            "high_risk_checks_added_count": self.high_risk_checks_added_count,
        }
