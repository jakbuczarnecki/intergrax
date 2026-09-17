# © Artur Czarnecki. All rights reserved.

"""Immutable per-assembly strategy snapshot (CE-02-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.budget.compaction import ContextCompactionStrategy, NoOpContextCompactionStrategy
from intergrax.context.budget.degradation import ContextDegradationPolicy, DefaultContextDegradationPolicy
from intergrax.context.budget.default_model_budget_policy import DefaultContextModelBudgetPolicy
from intergrax.context.budget.model_budget_policy import ContextModelBudgetPolicy
from intergrax.context.budget.token_counter import CharEstimateContextTokenCounter, ContextTokenCounter
from intergrax.context.registry import ContextPluginRegistry


@dataclass(frozen=True, slots=True)
class ContextAssemblyStrategySnapshot:
    model_budget_policy: ContextModelBudgetPolicy
    token_counter: ContextTokenCounter
    compaction_strategy: ContextCompactionStrategy
    degradation_policy: ContextDegradationPolicy


def snapshot_context_assembly_strategies(
    registry: ContextPluginRegistry,
    *,
    degradation_policy: ContextDegradationPolicy | None = None,
) -> ContextAssemblyStrategySnapshot:
    """Resolve replaceable strategies once per assembly."""
    degradation = registry.degradation_policy or degradation_policy or DefaultContextDegradationPolicy()
    return ContextAssemblyStrategySnapshot(
        model_budget_policy=registry.resolved_model_budget_policy(),
        token_counter=registry.resolved_token_counter(),
        compaction_strategy=registry.compaction_strategy,
        degradation_policy=degradation,
    )
