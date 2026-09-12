# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Quality-score strategy recommendation engine (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.strategy_recommendation.basis import (
    StrategyRecommendationBasis,
    StrategyRecommendationBasisKind,
)
from intergrax.contracts.self_healing.strategy_recommendation.confidence import (
    StrategyRecommendationConfidenceLevel,
)
from intergrax.contracts.self_healing.strategy_recommendation.context import (
    StrategyRecommendationCandidateQuality,
    StrategyRecommendationContext,
)
from intergrax.contracts.self_healing.strategy_recommendation.recommendation import StrategyRecommendation


def _rank_candidates(
    candidates: tuple[StrategyRecommendationCandidateQuality, ...],
) -> tuple[StrategyRecommendationCandidateQuality, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda row: (
                -row.assessment.quality_score,
                -row.assessment.execution_count,
                row.strategy_id,
            ),
        ),
    )


def _confidence_for_winner(
    winner: StrategyRecommendationCandidateQuality,
    runner_up: StrategyRecommendationCandidateQuality | None,
) -> StrategyRecommendationConfidenceLevel:
    if winner.assessment.execution_count == 0:
        return StrategyRecommendationConfidenceLevel.INSUFFICIENT_DATA
    if runner_up is None:
        return StrategyRecommendationConfidenceLevel.MEDIUM
    score_gap = winner.assessment.quality_score - runner_up.assessment.quality_score
    if winner.assessment.execution_count >= 3 and score_gap >= 0.1:
        return StrategyRecommendationConfidenceLevel.HIGH
    if score_gap > 0.0:
        return StrategyRecommendationConfidenceLevel.MEDIUM
    return StrategyRecommendationConfidenceLevel.LOW


def _basis_for_winner(
    winner: StrategyRecommendationCandidateQuality,
    runner_up: StrategyRecommendationCandidateQuality | None,
) -> StrategyRecommendationBasis:
    if winner.assessment.execution_count == 0:
        return StrategyRecommendationBasis(
            kind=StrategyRecommendationBasisKind.INSUFFICIENT_HISTORICAL_EVIDENCE,
            summary="No historical executions recorded for any candidate strategy",
        )
    if (
        runner_up is not None
        and winner.assessment.quality_score == runner_up.assessment.quality_score
        and winner.assessment.execution_count == runner_up.assessment.execution_count
    ):
        return StrategyRecommendationBasis(
            kind=StrategyRecommendationBasisKind.QUALITY_SCORE_TIE_BREAKER,
            summary="Equal historical quality score; stable strategy_id ordering applied",
        )
    return StrategyRecommendationBasis(
        kind=StrategyRecommendationBasisKind.HIGHEST_HISTORICAL_QUALITY_SCORE,
        summary="Highest historical quality score",
    )


@dataclass(frozen=True, slots=True)
class QualityBasedStrategyRecommendationEngine:
    """Ranks candidates using R5.2 ``StrategyQualityAssessment.quality_score`` only."""

    @property
    def engine_id(self) -> str:
        return "quality_based"

    def recommend(self, context: StrategyRecommendationContext) -> StrategyRecommendation:
        ranked_rows = _rank_candidates(context.candidates)
        winner = ranked_rows[0]
        runner_up = ranked_rows[1] if len(ranked_rows) > 1 else None
        ranked_ids = tuple(row.strategy_id for row in ranked_rows)
        return StrategyRecommendation(
            tenant_id=context.tenant_id,
            diagnostic_investigation_id=context.diagnostic_investigation_id,
            problem_id=context.problem_id,
            recommended_strategy_id=winner.strategy_id,
            ranked_strategy_ids=ranked_ids,
            basis=_basis_for_winner(winner, runner_up),
            confidence=_confidence_for_winner(winner, runner_up),
            supporting_assessment=winner.assessment,
            engine_id=self.engine_id,
        )


__all__ = ["QualityBasedStrategyRecommendationEngine"]
