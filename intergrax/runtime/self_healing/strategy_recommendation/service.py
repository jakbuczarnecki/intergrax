# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only strategy recommendation service (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.quality_evaluation.assessor import StrategyQualityAssessor
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria
from intergrax.contracts.self_healing.strategy_recommendation.context import (
    StrategyRecommendationCandidateQuality,
    StrategyRecommendationContext,
)
from intergrax.contracts.self_healing.strategy_recommendation.engine import StrategyRecommendationEngine
from intergrax.contracts.self_healing.strategy_recommendation.recommendation import StrategyRecommendation
from intergrax.contracts.self_healing.strategy_recommendation.request import StrategyRecommendationRequest

@dataclass(frozen=True, slots=True)
class StrategyRecommendationService:
    quality_evaluation: StrategyQualityAssessor
    engine: StrategyRecommendationEngine

    def recommend(self, request: StrategyRecommendationRequest) -> StrategyRecommendation:
        candidates: list[StrategyRecommendationCandidateQuality] = []
        for strategy_id in request.candidate_strategy_ids:
            assessment = self.quality_evaluation.assess(
                StrategyQualityEvaluationCriteria(
                    tenant_id=request.tenant_id,
                    strategy_id=strategy_id,
                ),
            )
            candidates.append(
                StrategyRecommendationCandidateQuality(
                    strategy_id=strategy_id,
                    assessment=assessment,
                ),
            )
        context = StrategyRecommendationContext(
            tenant_id=request.tenant_id,
            diagnostic_investigation_id=request.diagnostic_investigation_id,
            problem_id=request.problem_id,
            candidates=tuple(candidates),
        )
        return self.engine.recommend(context)


__all__ = ["StrategyRecommendationService"]
