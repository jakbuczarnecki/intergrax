# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform confidence evaluator (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.context import AdaptiveHealingContext
from intergrax.contracts.self_healing.adaptive.score import AdaptiveStrategyScore


class AdaptiveConfidenceEvaluator:
    """
    Aggregates confidence from strategy scores and context evidence.

    Implements bounded scoring — never fabricates confidence without evidence.
    """

    evaluator_id = "platform.adaptive.confidence"

    def evaluate_confidence(
        self,
        context: AdaptiveHealingContext,
        strategy_scores: tuple[AdaptiveStrategyScore, ...],
    ) -> tuple[float, tuple[str, ...], str]:
        evidence: list[str] = list(context.evidence_refs)
        for score in strategy_scores:
            evidence.extend(score.evidence_refs)
        evidence_tuple = tuple(dict.fromkeys(evidence))
        if not evidence_tuple:
            return (0.0, (), "confidence requires evidence_refs")
        if not strategy_scores:
            return (0.0, evidence_tuple, "no strategy scores to aggregate")
        weighted = sum(s.confidence * s.calculated_score for s in strategy_scores)
        total_score = sum(s.calculated_score for s in strategy_scores)
        if total_score <= 0.0:
            return (0.0, evidence_tuple, "zero aggregate strategy score")
        overall = max(0.0, min(1.0, weighted / total_score))
        explanation = (
            f"aggregated from {len(strategy_scores)} strategy score(s) "
            f"with {len(evidence_tuple)} evidence ref(s)"
        )
        return (overall, evidence_tuple, explanation)


__all__ = ["AdaptiveConfidenceEvaluator"]
