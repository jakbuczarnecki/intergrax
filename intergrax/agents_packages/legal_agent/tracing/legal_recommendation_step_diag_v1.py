# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LegalRecommendationStepDiagV1(DiagnosticPayload):
    step_name: str
    clauses_context_count: int
    legal_checks_count: int
    sensitive_flags_count: int
    policy_violations_count: int
    recommendations_added_count: int
    high_priority_recommendations_count: int

    def redact(self) -> LegalRecommendationStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.recommendation.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "clauses_context_count": self.clauses_context_count,
            "legal_checks_count": self.legal_checks_count,
            "sensitive_flags_count": self.sensitive_flags_count,
            "policy_violations_count": self.policy_violations_count,
            "recommendations_added_count": self.recommendations_added_count,
            "high_priority_recommendations_count": self.high_priority_recommendations_count,
        }
