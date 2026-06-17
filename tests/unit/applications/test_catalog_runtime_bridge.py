# © Artur Czarnecki. All rights reserved.

"""TS-1: Tool/skill catalog profiles → RuntimeConfig bridge."""

from __future__ import annotations

import pytest

from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications._shared.catalog_runtime_bridge import (
    apply_catalog_profiles_from_build_context,
    apply_catalog_profiles_from_environment,
    apply_skill_profile_to_runtime_config,
    apply_tool_engine_settings_from_environment,
    apply_tool_profile_to_runtime_config,
)
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-ts",
        agent_id="echo",
        user_id="user-ts",
        session_id="session-ts",
        message="catalog bridge probe",
    )


def test_apply_tool_profile_to_runtime_config() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    profile = ToolProfile(enabled=("tool.echo",))

    apply_tool_profile_to_runtime_config(config, profile)

    assert config.tool_profile is profile
    assert list(config.tool_profile.enabled) == ["tool.echo"]


def test_apply_skill_profile_to_runtime_config() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    profile = SkillProfile(enabled_bundles=("harness.stack",))

    apply_skill_profile_to_runtime_config(config, profile)

    assert config.skill_profile is profile
    assert list(config.skill_profile.enabled_bundles) == ["harness.stack"]


def test_apply_catalog_profiles_from_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "tool_profile": ToolProfile(enabled=("tool.a",)),
            "skill_profile": SkillProfile(enabled_bundles=("bundle.a",)),
        }
    )
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_catalog_profiles_from_environment(config, env)

    assert config.tool_profile is not None
    assert list(config.tool_profile.enabled) == ["tool.a"]
    assert config.skill_profile is not None
    assert list(config.skill_profile.enabled_bundles) == ["bundle.a"]


def test_apply_catalog_profiles_from_build_context_overrides_environment() -> None:
    env_profile = ToolProfile(enabled=("tool.env",))
    wired_profile = ToolProfile(enabled=("tool.wired",))
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    apply_tool_profile_to_runtime_config(config, env_profile)

    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    build_ctx = ApplicationBuildContext.for_manifest(
        build_lab_manifest(settings),
        settings=settings,
        tool_profile=wired_profile,
        skill_profile=SkillProfile(enabled_bundles=("wired.bundle",)),
    )
    apply_catalog_profiles_from_build_context(config, build_ctx)

    assert config.tool_profile is wired_profile
    assert config.skill_profile is not None
    assert list(config.skill_profile.enabled_bundles) == ["wired.bundle"]


def test_materialize_runtime_config_includes_catalog_profiles() -> None:
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    config = materialize_runtime_config(
        _request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.tool_profile is not None
    assert config.skill_profile is not None
    assert config.tool_wiring_context is wiring.build_context.tool_wiring_context


def test_apply_tool_engine_settings_from_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"reasoning_profile": ReasoningProfile(tool_planner_prompt_id="tools_custom")}
    )
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_tool_engine_settings_from_environment(config, env)

    assert config.tool_planner_prompt_id == "tools_custom"
    assert config.engine_planner_prompt_id == "planner_default"


def test_apply_tool_engine_settings_engine_planner_prompt() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reasoning_profile": ReasoningProfile(engine_planner_prompt_id="planner_replan_default"),
        }
    )
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_tool_engine_settings_from_environment(config, env)

    assert config.engine_planner_prompt_id == "planner_replan_default"


def test_materialize_runtime_config_lab_harness_uses_environment_catalogs() -> None:
    env = build_lab_environment_profile(LabApplicationSettings.from_env())
    harness = default_reference_harness()
    config = materialize_runtime_config(
        _request(),
        harness,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.tool_profile is env.tool_profile
    assert config.skill_profile is env.skill_profile
