# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile memory/context fields to RuntimeConfig (Phase MEM-1)."""

from __future__ import annotations

from intergrax.applications._shared.context_runtime_bridge import (
    apply_context_profile_to_runtime_config,
    apply_context_profiles_from_environment,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig

__all__ = [
    "apply_context_profile_to_runtime_config",
    "apply_memory_profile_to_runtime_config",
    "apply_environment_profiles_to_runtime_config",
]


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


def apply_environment_profiles_to_runtime_config(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Full memory + context bridge from environment profile."""
    apply_memory_profile_to_runtime_config(config, env.memory_profile)
    apply_context_profiles_from_environment(config, env)
    return config
