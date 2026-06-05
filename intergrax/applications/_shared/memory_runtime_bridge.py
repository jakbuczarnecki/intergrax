# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile memory/context fields to RuntimeConfig (Phase MEM-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
    MemoryProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_memory_profile_to_runtime_config(
    config: RuntimeConfig,
    memory: MemoryProfile,
) -> RuntimeConfig:
    """Apply ``MemoryProfile`` toggles and policy fields."""
    config.enable_user_profile_memory = memory.enable_user_memory
    config.enable_org_profile_memory = memory.enable_org_memory
    config.enable_user_longterm_memory = memory.enable_long_term_memory
    config.enable_task_memory = memory.enable_task_memory
    config.memory_retention_days = memory.retention_days
    config.memory_scope_boundary = memory.scope_boundary
    return config


def apply_context_profile_to_runtime_config(
    config: RuntimeConfig,
    context: ContextProfile,
) -> RuntimeConfig:
    """Apply ``ContextProfile`` budget and assembly options."""
    config.enable_rag = context.enable_rag
    config.enable_websearch = context.enable_websearch
    if context.budget_policy is not None:
        config.context_budget_policy = context.budget_policy
    config.task_context_assembly_options = context.assembly_options
    config.context_decision_profile = context.decision.model_dump(mode="json")
    if context.decision.max_memory_entries_in_context != config.max_longterm_entries_per_query:
        config.max_longterm_entries_per_query = context.decision.max_memory_entries_in_context
    return config


def apply_environment_profiles_to_runtime_config(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Full memory + context bridge from environment profile."""
    apply_memory_profile_to_runtime_config(config, env.memory_profile)
    apply_context_profile_to_runtime_config(config, env.context_profile)
    if config.context_budget_policy is not None and config.run_budget is None:
        from intergrax.runtime.nexus.budget.budget_models import RunBudget

        policy = config.context_budget_policy
        config.run_budget = RunBudget(max_total_tokens=policy.max_tokens_estimate)
    return config
