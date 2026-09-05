# © Artur Czarnecki. All rights reserved.

"""Tests for skill tool requirement validation at application composition."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring
from intergrax.applications._shared.skill_tool_profile import (
    assert_skill_tool_requirements_for_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.registry.tool_requirements import SkillToolRequirementError
from intergrax.applications.contracts.capability_dependency import (
    RequiredCapabilityDependencyUnavailableError,
)
from intergrax.tools.registry.profile import ToolProfile
from lab_application.host.settings import LabApplicationSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_assert_skill_tool_requirements_rejects_missing_tools() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(
            skill_id="research.web_evidence",
            description="web evidence",
            tool_ids=("websearch.query",),
        ),
    )
    tool_profile = ToolProfile(enabled=["rag.retrieve"])

    with pytest.raises(SkillToolRequirementError) as exc_info:
        assert_skill_tool_requirements_for_profile(
            tool_profile,
            SkillProfile(enabled=["research.web_evidence"]),
            skill_registry=registry,
        )

    violation = exc_info.value.resolution.violations[0]
    assert violation.skill_id == "research.web_evidence"
    assert violation.tool_id == "websearch.query"


def test_skill_requirements_do_not_expand_host_tool_profile() -> None:
    tool_profile = ToolProfile(enabled=["tool.a"])
    original_enabled = list(tool_profile.enabled)

    with pytest.raises(SkillToolRequirementError):
        assert_skill_tool_requirements_for_profile(
            tool_profile,
            SkillProfile(enabled=["skill.missing"]),
            skill_registry=_registry_requiring("skill.missing", "tool.b"),
        )

    assert tool_profile.enabled == original_enabled
    assert "tool.b" not in tool_profile.enabled


def test_empty_skill_profile_is_noop() -> None:
    base = ToolProfile(enabled=["database.query"])
    resolution = assert_skill_tool_requirements_for_profile(base, SkillProfile())
    assert resolution.is_satisfied is True
    assert resolution.required_tool_ids == ()


def test_wire_application_environment_rejects_unsatisfied_skill_tools() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "tool_profile": ToolProfile(enabled=["rag.retrieve"]),
            "skill_profile": SkillProfile(enabled_bundles=["legal"]),
        },
    )
    manifest = ApplicationManifest(
        app_id="skill_tool_guard",
        name="Skill Tool Guard",
        route_prefix="/v1/skill_tool_guard",
        env_prefix="SKILL_TOOL_GUARD_",
        agents=[],
    )

    with pytest.raises(RequiredCapabilityDependencyUnavailableError):
        wire_application_environment(manifest, env, conformance_check=False)


def test_lab_environment_declares_tool_availability_for_skills() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    assert env.tool_profile.register_all_catalog_bundles is False
    assert env.tool_profile.enabled_bundles

    skill_wiring = build_application_skill_wiring(env.skill_profile)
    resolution = assert_skill_tool_requirements_for_profile(
        env.tool_profile,
        env.skill_profile,
        skill_registry=skill_wiring.registry,
    )

    assert resolution.is_satisfied


def _registry_requiring(skill_id: str, tool_id: str) -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(skill_id=skill_id, description=skill_id, tool_ids=(tool_id,)),
    )
    return registry
