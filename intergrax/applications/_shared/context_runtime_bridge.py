# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile context fields to RuntimeConfig (Phase CTX-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_context_profile_to_runtime_config(
    config: RuntimeConfig,
    context: ContextProfile,
) -> RuntimeConfig:
    """Apply ``ContextProfile`` budget, assembly, and decision fields."""
    config.enable_rag = context.enable_rag
    config.enable_websearch = context.enable_websearch
    if context.budget_policy is not None:
        config.context_budget_policy = context.budget_policy
    config.task_context_assembly_options = context.assembly_options
    config.context_decision_profile = context.decision.model_dump(mode="json")
    if context.decision.max_memory_entries_in_context != config.max_longterm_entries_per_query:
        config.max_longterm_entries_per_query = context.decision.max_memory_entries_in_context
    derive_run_budget_from_context_policy(config)
    return config


def derive_run_budget_from_context_policy(config: RuntimeConfig) -> RuntimeConfig:
    """Mirror context token budget into Nexus ``RunBudget`` when unset."""
    if config.context_budget_policy is not None and config.run_budget is None:
        from intergrax.runtime.nexus.budget.budget_models import RunBudget

        policy = config.context_budget_policy
        config.run_budget = RunBudget(max_total_tokens=policy.max_tokens_estimate)
    return config


def apply_context_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared context profile."""
    return apply_context_profile_to_runtime_config(config, env.context_profile)
