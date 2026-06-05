# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile prompt catalog to RuntimeConfig (Phase PE-1)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PromptProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


def apply_prompt_profile_to_runtime_config(
    config: RuntimeConfig,
    prompt: PromptProfile,
) -> RuntimeConfig:
    """Attach prompt catalog path for runtime context resolution."""
    if prompt.catalog_path is not None:
        config.prompt_catalog_path = str(prompt.catalog_path)
    return config


def apply_prompt_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared prompt profile."""
    return apply_prompt_profile_to_runtime_config(config, env.prompt_profile)
