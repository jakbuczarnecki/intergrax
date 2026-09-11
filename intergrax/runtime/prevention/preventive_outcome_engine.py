# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation outcome learning — extends R5 without new authority (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.predictive.analyzer_quality_profile import profile_after_outcome_counts
from intergrax.contracts.preventive.outcome_evaluation import (
    RecommendationEffectiveness,
    RecommendationOutcomeEvaluation,
)
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)


@dataclass
class InMemoryPreventiveRecommendationOutcomeStore:
    evaluations: list[RecommendationOutcomeEvaluation] = field(default_factory=list)

    def append(self, evaluation: RecommendationOutcomeEvaluation) -> RecommendationOutcomeEvaluation:
        self.evaluations.append(evaluation)
        return evaluation

    def historical_success_rate(self, *, tenant_id: str, analyzer_id: str) -> float:
        scoped = [
            item
            for item in self.evaluations
            if item.tenant_id == tenant_id and item.analyzer_id == analyzer_id
        ]
        if not scoped:
            return 0.7
        successes = sum(
            1
            for item in scoped
            if item.effectiveness is RecommendationEffectiveness.TRUE_PREVENTION
        )
        return successes / len(scoped)


@dataclass
class PreventiveOutcomeEngine:
    store: InMemoryPreventiveRecommendationOutcomeStore = field(
        default_factory=InMemoryPreventiveRecommendationOutcomeStore,
    )
    quality_store: InMemoryPredictiveAnalyzerQualityStore | None = None

    def record(self, evaluation: RecommendationOutcomeEvaluation) -> RecommendationOutcomeEvaluation:
        stored = self.store.append(evaluation)
        if (
            self.quality_store is not None
            and evaluation.effectiveness is RecommendationEffectiveness.TRUE_PREVENTION
        ):
            profile = self.quality_store.get_profile(
                tenant_id=evaluation.tenant_id,
                analyzer_id=evaluation.analyzer_id,
            )
            updated = profile_after_outcome_counts(
                profile,
                predictions=profile.predictions + 1,
                true_positive=profile.true_positive + 1,
                false_positive=profile.false_positive,
                false_negative=profile.false_negative,
                true_negative=profile.true_negative,
            )
            self.quality_store.put_profile(updated)
        return stored


__all__ = ["InMemoryPreventiveRecommendationOutcomeStore", "PreventiveOutcomeEngine"]
