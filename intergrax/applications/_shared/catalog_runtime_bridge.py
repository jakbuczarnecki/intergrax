# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile tool/skill catalogs to RuntimeConfig (Phase TS-1)."""

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile


def apply_tool_profile_to_runtime_config(
    config: RuntimeConfig,
    tool_profile: ToolProfile,
) -> RuntimeConfig:
    """Attach resolved ``ToolProfile`` to runtime config."""
    config.tool_profile = tool_profile
    return config


def apply_skill_profile_to_runtime_config(
    config: RuntimeConfig,
    skill_profile: SkillProfile,
) -> RuntimeConfig:
    """Attach resolved ``SkillProfile`` to runtime config."""
    config.skill_profile = skill_profile
    return config


def apply_tool_engine_settings_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Bridge reasoning/tool engine settings before ``RuntimeContext.build`` (TOOL-ENG-0)."""
    from intergrax.runtime.nexus.config_types import ToolSelectionMode

    config.tool_planner_prompt_id = env.reasoning_profile.tool_planner_prompt_id
    try:
        config.tool_selection_mode = ToolSelectionMode(env.tool_selection_mode)
    except ValueError:
        config.tool_selection_mode = ToolSelectionMode.STATIC
    config.tool_selection_top_k = env.tool_selection_top_k
    return config


def apply_catalog_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared tool and skill profiles."""
    apply_tool_profile_to_runtime_config(config, env.tool_profile)
    apply_skill_profile_to_runtime_config(config, env.skill_profile)
    apply_tool_engine_settings_from_environment(config, env)
    return config


def apply_catalog_profiles_from_build_context(
    config: RuntimeConfig,
    build_ctx: ApplicationBuildContext,
) -> RuntimeConfig:
    """
    Overlay wired catalog artifacts from Tier-3 bootstrap.

    Wired profiles (sandbox-adjusted tools, resolved registries) take precedence
    over raw environment defaults.
    """
    if build_ctx.tool_profile is not None:
        config.tool_profile = build_ctx.tool_profile
    if build_ctx.tool_wiring_context is not None:
        config.tool_wiring_context = build_ctx.tool_wiring_context
    if build_ctx.skill_profile is not None:
        config.skill_profile = build_ctx.skill_profile
    return config
