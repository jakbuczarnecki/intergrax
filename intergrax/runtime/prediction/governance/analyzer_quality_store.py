# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tenant-scoped analyzer quality profiles (PREDICTIVE R4, R5 learning)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.predictive.analyzer_quality_profile import (
    PredictiveAnalyzerQualityProfile,
    default_analyzer_quality_profile,
    profile_after_outcome_counts,
)
from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation
from intergrax.contracts.predictive.outcome.types import PredictionOutcomeType


@dataclass
class InMemoryPredictiveAnalyzerQualityStore:
    """In-process profile store — keyed by tenant and analyzer."""

    _profiles: dict[tuple[str, str], PredictiveAnalyzerQualityProfile] = field(
        default_factory=dict,
    )

    def get_profile(self, *, tenant_id: str, analyzer_id: str) -> PredictiveAnalyzerQualityProfile:
        key = (tenant_id, analyzer_id)
        if key not in self._profiles:
            self._profiles[key] = default_analyzer_quality_profile(
                analyzer_id=analyzer_id,
                tenant_id=tenant_id,
            )
        return self._profiles[key]

    def put_profile(self, profile: PredictiveAnalyzerQualityProfile) -> None:
        self._profiles[(profile.tenant_id, profile.analyzer_id)] = profile


def apply_outcome_evaluation(
    store: InMemoryPredictiveAnalyzerQualityStore,
    evaluation: PredictionOutcomeEvaluation,
) -> PredictiveAnalyzerQualityProfile:
    profile = store.get_profile(
        tenant_id=evaluation.tenant_id,
        analyzer_id=evaluation.analyzer_id,
    )
    predictions = profile.predictions + 1
    true_positive = profile.true_positive
    false_positive = profile.false_positive
    false_negative = profile.false_negative
    true_negative = profile.true_negative

    if evaluation.outcome_type is PredictionOutcomeType.TRUE_POSITIVE:
        true_positive += 1
    elif evaluation.outcome_type is PredictionOutcomeType.FALSE_POSITIVE:
        false_positive += 1
    elif evaluation.outcome_type is PredictionOutcomeType.NO_INCIDENT:
        true_negative += 1
    elif evaluation.outcome_type is PredictionOutcomeType.INCIDENT_OCCURRED:
        false_negative += 1

    updated = profile_after_outcome_counts(
        profile,
        predictions=predictions,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )
    store.put_profile(updated)
    return updated


__all__ = ["InMemoryPredictiveAnalyzerQualityStore", "apply_outcome_evaluation"]
