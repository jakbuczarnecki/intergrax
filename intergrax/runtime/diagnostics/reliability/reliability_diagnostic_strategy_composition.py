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


__all__ = [
    "build_reliability_case_default_grouping_strategy",
    "default_reliability_diagnostic_reconciliation_policies",
    "register_reliability_case_default_grouping_strategy",
]
