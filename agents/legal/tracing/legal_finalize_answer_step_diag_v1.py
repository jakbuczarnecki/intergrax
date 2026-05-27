# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

LegalFinalizeAnswerOutcome = Literal["ok", "empty_fallback"]


@dataclass(frozen=True)
class LegalFinalizeAnswerStepDiagV1(DiagnosticPayload):
    step_name: str
    outcome: LegalFinalizeAnswerOutcome
    answer_length_chars: int
    used_rag: bool
    used_attachments_context: bool
    clauses_count: int
    legal_checks_count: int
    sensitive_flags_count: int
    compliance_results_count: int
    uncertainties_count: int
    policy_violations_count: int
    recommendations_count: int
    decision_status: Optional[str]
    decision_enforcement_modified: bool

    def redact(self) -> LegalFinalizeAnswerStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.finalize_answer.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "outcome": self.outcome,
            "answer_length_chars": self.answer_length_chars,
            "used_rag": self.used_rag,
            "used_attachments_context": self.used_attachments_context,
            "clauses_count": self.clauses_count,
            "legal_checks_count": self.legal_checks_count,
            "sensitive_flags_count": self.sensitive_flags_count,
            "compliance_results_count": self.compliance_results_count,
            "uncertainties_count": self.uncertainties_count,
            "policy_violations_count": self.policy_violations_count,
            "recommendations_count": self.recommendations_count,
            "decision_status": self.decision_status,
            "decision_enforcement_modified": self.decision_enforcement_modified,
        }
