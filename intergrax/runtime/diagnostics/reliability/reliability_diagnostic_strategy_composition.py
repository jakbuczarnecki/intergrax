# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Explicit composition for ERL reliability diagnostic grouping strategies."""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingStrategyRegistry
from intergrax.runtime.diagnostics.problem_lifecycle import (
    DeterministicProblemReconciliationPolicy,
    ProblemReconciliationPolicy,
)
from intergrax.contracts.enterprise_reliability.diagnostics.grouping import (
    ExternalEffectReliabilityProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_default_grouping_strategy import (
    ReliabilityCaseDefaultGroupingStrategy,
    ReliabilityCaseDefaultObservationGroupingStrategy,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_grouping_reconciliation import (
    ReliabilityCaseProblemReconciliationPolicy,
)
from intergrax.contracts.enterprise_reliability.diagnostics.classification import (
    ExternalEffectReliabilityDiagnosticRecommendationStrategy,
    ExternalEffectReliabilityDiagnosticSeverityStrategy,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_recommendation_strategy import (
    ConservativeReliabilityRecommendationStrategy,
)
from intergrax.runtime.diagnostics.reliability.conservative_reliability_severity_strategy import (
    ConservativeReliabilitySeverityStrategy,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_classification_service import (
    ReliabilityDiagnosticClassificationService,
)


def default_reliability_diagnostic_reconciliation_policies() -> tuple[
    ProblemReconciliationPolicy,
    ...
]:
    return (
        DeterministicProblemReconciliationPolicy(),
        ReliabilityCaseProblemReconciliationPolicy(),
    )


def build_reliability_case_default_grouping_strategy(
    *,
    observation_grouping: ExternalEffectReliabilityProblemGroupingStrategy | None = None,
) -> tuple[
    ReliabilityCaseDefaultGroupingStrategy,
    ExternalEffectReliabilityProblemGroupingStrategy,
]:
    resolved_observation_grouping = (
        observation_grouping or ReliabilityCaseDefaultObservationGroupingStrategy()
    )
    batch_strategy = ReliabilityCaseDefaultGroupingStrategy(
        observation_grouping=resolved_observation_grouping,
    )
    return batch_strategy, resolved_observation_grouping


def register_reliability_case_default_grouping_strategy(
    registry: ProblemGroupingStrategyRegistry,
    *,
    strategy: ReliabilityCaseDefaultGroupingStrategy | None = None,
    observation_grouping: ExternalEffectReliabilityProblemGroupingStrategy | None = None,
) -> ReliabilityCaseDefaultGroupingStrategy:
    if strategy is None:
        strategy, _ = build_reliability_case_default_grouping_strategy(
            observation_grouping=observation_grouping,
        )
    registry.register(strategy)
    return strategy


def build_reliability_diagnostic_classification_service(
    *,
    severity_strategy: ExternalEffectReliabilityDiagnosticSeverityStrategy | None = None,
    recommendation_strategy: ExternalEffectReliabilityDiagnosticRecommendationStrategy | None = None,
) -> ReliabilityDiagnosticClassificationService:
    resolved_severity = severity_strategy or ConservativeReliabilitySeverityStrategy()
    resolved_recommendation = recommendation_strategy or ConservativeReliabilityRecommendationStrategy()
    return ReliabilityDiagnosticClassificationService(
        severity_strategy=resolved_severity,
        recommendation_strategy=resolved_recommendation,
        severity_fallback=ConservativeReliabilitySeverityStrategy(),
        recommendation_fallback=ConservativeReliabilityRecommendationStrategy(),
    )


__all__ = [
    "build_reliability_case_default_grouping_strategy",
    "build_reliability_diagnostic_classification_service",
    "default_reliability_diagnostic_reconciliation_policies",
    "register_reliability_case_default_grouping_strategy",
]
