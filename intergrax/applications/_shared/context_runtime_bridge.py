# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile context fields to RuntimeConfig (Phase CTX-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig

CONTEXT_ENGINE_PROFILE_METADATA_KEY = "context_engine_profile.v1"


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
    if context.drift_monitoring_enabled:
        config.metadata["context_drift_monitoring.v1"] = {
            "enabled": True,
            "alert_threshold": context.drift_alert_threshold,
        }
    if context.semantic_compression_enabled:
        config.metadata["semantic_compression.v1"] = {
            "enabled": True,
            "strategy": context.default_history_compression,
        }
    if context.decision.max_memory_entries_in_context != config.max_longterm_entries_per_query:
        config.max_longterm_entries_per_query = context.decision.max_memory_entries_in_context
    config.metadata[CONTEXT_ENGINE_PROFILE_METADATA_KEY] = {
        "engine_preset": context.engine_preset,
        "engine_ref": context.engine_ref,
        "context_plugin_ids": list(context.context_plugin_ids),
    }
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
