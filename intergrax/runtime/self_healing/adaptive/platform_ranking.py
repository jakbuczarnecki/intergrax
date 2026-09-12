# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default strategy ranking provider (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.context import AdaptiveHealingContext
from intergrax.contracts.self_healing.adaptive.score import AdaptiveScoringFactor, AdaptiveStrategyScore


class PlatformHistoricalRankingProvider:
    provider_id = "platform.adaptive.historical_ranking"

    def rank_strategies(self, context: AdaptiveHealingContext) -> tuple[AdaptiveStrategyScore, ...]:
        outcome_map = {o.strategy_id: o for o in context.historical_outcomes}
        validation_map = {v.strategy_id: v for v in context.validation_quality}
        rollback_map = {r.strategy_id: r for r in context.rollback_history}
        scores: list[AdaptiveStrategyScore] = []
        for strategy in context.strategy_candidates:
            sid = strategy.strategy_id
            outcome = outcome_map.get(sid)
            validation = validation_map.get(sid)
            rollback = rollback_map.get(sid)
            success = 0.5
            if outcome is not None:
                total = outcome.successful_preventions + outcome.failed_actions
                if total > 0:
                    success = outcome.successful_preventions / total
            val_q = validation.quality_score if validation is not None else 0.5
            rollback_penalty = 0.0
            if rollback is not None and rollback.rollback_count > 0:
                rollback_penalty = min(1.0, rollback.rollback_count * 0.1)
            similarity_boost = 0.05 if context.context_similarity_refs else 0.0
            calculated = max(0.0, min(1.0, (success * 0.4) + (val_q * 0.35) - (rollback_penalty * 0.2) + similarity_boost))
            evidence: list[str] = list(context.evidence_refs)
            if validation is not None:
                evidence.extend(validation.evidence_refs)
            if rollback is not None:
                evidence.extend(rollback.evidence_refs)
            evidence.extend(context.context_similarity_refs)
            evidence_tuple = tuple(dict.fromkeys(evidence))
            confidence = calculated if evidence_tuple else 0.0
            factors = (
                AdaptiveScoringFactor("historical_success", 0.4, success),
                AdaptiveScoringFactor("validation_quality", 0.35, val_q),
                AdaptiveScoringFactor("rollback_frequency", 0.2, rollback_penalty),
                AdaptiveScoringFactor("context_similarity", 0.05, similarity_boost),
            )
            scores.append(
                AdaptiveStrategyScore(
                    strategy_id=sid,
                    calculated_score=calculated,
                    confidence=confidence,
                    evidence_refs=evidence_tuple,
                    scoring_factors=factors,
                ),
            )
        return tuple(scores)


__all__ = ["PlatformHistoricalRankingProvider"]
