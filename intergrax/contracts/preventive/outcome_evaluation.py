# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation effectiveness feedback loop (PREVENTIVE R6, extends R5 learning)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class OperatorRecommendationDecision(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class RecommendationEffectiveness(StrEnum):
    TRUE_PREVENTION = "TRUE_PREVENTION"
    INEFFECTIVE = "INEFFECTIVE"
    UNKNOWN = "UNKNOWN"
    OPERATOR_REJECTED = "OPERATOR_REJECTED"


@dataclass(frozen=True, slots=True)
class RecommendationOutcomeEvaluation:
    recommendation_id: str
    tenant_id: str
    analyzer_id: str
    operator_decision: OperatorRecommendationDecision
    effectiveness: RecommendationEffectiveness
    evidence_refs: tuple[str, ...]
    evaluated_at: datetime
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.recommendation_id.strip():
            raise ValueError("recommendation_id required")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id required")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


__all__ = [
    "OperatorRecommendationDecision",
    "RecommendationEffectiveness",
    "RecommendationOutcomeEvaluation",
]
