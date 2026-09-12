# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action outcome → prediction quality (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.predictive.analyzer_quality_profile import profile_after_outcome_counts
from intergrax.contracts.preventive.actions.outcome import (
    PreventiveActionObservedOutcome,
    PreventiveActionOutcomeEvaluation,
)
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)


@dataclass
class InMemoryPreventiveActionOutcomeStore:
    evaluations: list[PreventiveActionOutcomeEvaluation] = field(default_factory=list)

    def append(
        self,
        evaluation: PreventiveActionOutcomeEvaluation,
    ) -> PreventiveActionOutcomeEvaluation:
        self.evaluations.append(evaluation)
        return evaluation


@dataclass
class PreventiveActionOutcomeEngine:
    store: InMemoryPreventiveActionOutcomeStore = field(
        default_factory=InMemoryPreventiveActionOutcomeStore,
    )
    quality_store: InMemoryPredictiveAnalyzerQualityStore | None = None

    def record(
        self,
        evaluation: PreventiveActionOutcomeEvaluation,
    ) -> PreventiveActionOutcomeEvaluation:
        stored = self.store.append(evaluation)
        if (
            self.quality_store is not None
            and evaluation.observed is PreventiveActionObservedOutcome.INCIDENT_AVOIDED
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


__all__ = ["InMemoryPreventiveActionOutcomeStore", "PreventiveActionOutcomeEngine"]
