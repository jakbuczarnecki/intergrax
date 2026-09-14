# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Explicit composition for ERL reliability diagnostic grouping strategies."""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingStrategyRegistry
from intergrax.runtime.diagnostics.problem_lifecycle import (
    DeterministicProblemReconciliationPolicy,
    ProblemReconciliationPolicy,
)
from intergrax.runtime.diagnostics.reliability.reliability_case_default_grouping_strategy import (
    ReliabilityCaseDefaultGroupingStrategy,
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


def register_reliability_case_default_grouping_strategy(
    registry: ProblemGroupingStrategyRegistry,
    *,
    strategy: ReliabilityCaseDefaultGroupingStrategy | None = None,
) -> ReliabilityCaseDefaultGroupingStrategy:
    resolved = strategy or ReliabilityCaseDefaultGroupingStrategy()
    registry.register(resolved)
    return resolved


__all__ = [
    "default_reliability_diagnostic_reconciliation_policies",
    "register_reliability_case_default_grouping_strategy",
]
