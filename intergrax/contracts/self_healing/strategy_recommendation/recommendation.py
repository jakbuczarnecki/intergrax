# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy recommendation result — advisory only (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.strategy_recommendation.basis import StrategyRecommendationBasis
from intergrax.contracts.self_healing.strategy_recommendation.confidence import (
    StrategyRecommendationConfidenceLevel,
)


@dataclass(frozen=True, slots=True)
class StrategyRecommendation:
    """
    Suggested strategy for a diagnostic situation.

    Contains no executors, workflow handles, or execution directives.
    """

    tenant_id: str
    diagnostic_investigation_id: str
    problem_id: str
    recommended_strategy_id: str
    ranked_strategy_ids: tuple[str, ...]
    basis: StrategyRecommendationBasis
    confidence: StrategyRecommendationConfidenceLevel
    supporting_assessment: StrategyQualityAssessment
    engine_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if not self.problem_id.strip():
            raise ValueError("problem_id required")
        if not self.recommended_strategy_id.strip():
            raise ValueError("recommended_strategy_id required")
        if not self.ranked_strategy_ids:
            raise ValueError("ranked_strategy_ids must be non-empty")
        if self.recommended_strategy_id not in self.ranked_strategy_ids:
            raise ValueError("recommended_strategy_id must appear in ranked_strategy_ids")
        if self.ranked_strategy_ids[0] != self.recommended_strategy_id:
            raise ValueError("recommended_strategy_id must be first in ranked_strategy_ids")
        if not self.engine_id.strip():
            raise ValueError("engine_id required")
        if self.supporting_assessment.strategy_id != self.recommended_strategy_id:
            raise ValueError("supporting_assessment.strategy_id mismatch")
        if self.supporting_assessment.tenant_id != self.tenant_id:
            raise ValueError("tenant isolation violation: supporting_assessment")


__all__ = ["StrategyRecommendation"]
