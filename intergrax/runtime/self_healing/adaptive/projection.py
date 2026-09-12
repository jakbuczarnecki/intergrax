# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic read projections for adaptive healing (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.recommendation import AdaptiveHealingRecommendation
from intergrax.contracts.self_healing_investigation_read import AdaptiveHealingInsightView


def project_adaptive_healing_insights(
    recommendation: AdaptiveHealingRecommendation | None,
) -> tuple[AdaptiveHealingInsightView, ...]:
    if recommendation is None:
        return ()
    ranking_lines = tuple(
        f"{score.strategy_id}:{score.calculated_score:.3f}" for score in recommendation.strategy_scores
    )
    return (
        AdaptiveHealingInsightView(
            tenant_id=recommendation.tenant_id,
            status=recommendation.status.value,
            overall_confidence=recommendation.overall_confidence,
            recommended_strategy_order=recommendation.recommended_strategy_order,
            strategy_ranking_summary=ranking_lines,
            confidence_explanation=recommendation.confidence_explanation,
            adaptive_insights=recommendation.adaptive_insights,
            evidence_refs=recommendation.evidence_refs,
        ),
    )


__all__ = ["project_adaptive_healing_insights"]
