# © Artur Czarnecki. All rights reserved.

"""Pluggable adaptive decision intelligence contracts (DS-E2E-15J-L10)."""

from __future__ import annotations

from typing import Protocol

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDecisionIntelligenceContext,
    AdaptiveDecisionIntelligenceInput,
    AdaptiveDecisionRecommendation,
    AdaptiveReasoningInsight,
    HistoricalEvidence,
)


class AdaptiveDecisionContextProvider(Protocol):
    """Pluggable source of historical and situational evidence."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]: ...


class AdaptiveReasoningProvider(Protocol):
    """Pluggable analysis over merged context — returns insights, not actions."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def reason(
        self,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveReasoningInsight, ...]: ...


class AdaptiveRecommendationProvider(Protocol):
    """Pluggable presentation of recommendations from reasoning insights."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def recommend(
        self,
        reasoning_insights: tuple[AdaptiveReasoningInsight, ...],
        *,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveDecisionRecommendation, ...]: ...


__all__ = [
    "AdaptiveDecisionContextProvider",
    "AdaptiveReasoningProvider",
    "AdaptiveRecommendationProvider",
]
