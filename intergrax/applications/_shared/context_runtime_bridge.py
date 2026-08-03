# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile context fields to RuntimeConfig (Phase CTX-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.wiring.context_runtime_bridge import (
    CONTEXT_ENGINE_PROFILE_METADATA_KEY,
    apply_context_profile_to_runtime_config as _apply_context_profile_to_runtime_config,
    derive_run_budget_from_context_policy,
)

__all__ = [
    "CONTEXT_ENGINE_PROFILE_METADATA_KEY",
    "apply_context_profile_to_runtime_config",
    "apply_context_profiles_from_environment",
    "derive_run_budget_from_context_policy",
]


def apply_context_profile_to_runtime_config(
    config: RuntimeConfig,
    context: ContextProfile,
) -> RuntimeConfig:
    """Apply environment ``ContextProfile`` via the canonical runtime bridge."""
    return _apply_context_profile_to_runtime_config(config, context)


def apply_context_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared context profile."""
    return apply_context_profile_to_runtime_config(config, env.context_profile)
