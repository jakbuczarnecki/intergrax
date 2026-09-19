# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile integration selection to RuntimeConfig (Phase INT-1)."""

from __future__ import annotations

from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_integration_profile_to_runtime_config(
    config: RuntimeConfig,
    integration_profile: IntegrationProfile,
) -> RuntimeConfig:
    """Attach resolved ``IntegrationProfile`` to runtime config."""
    config.integration_profile = integration_profile
    return config


def apply_integration_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared integration profile."""
    return apply_integration_profile_to_runtime_config(config, env.integration_profile)


def apply_integration_profiles_from_composition(
    config: RuntimeConfig,
    composition: ApplicationCompositionContext,
) -> RuntimeConfig:
    """Overlay wired integration profile from Tier-3 bootstrap."""
    if composition.integration_profile is not None:
        config.integration_profile = composition.integration_profile
    return config
