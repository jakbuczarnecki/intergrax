# © Artur Czarnecki. All rights reserved.

"""Tests for skill tool requirement resolution (P0-SAFETY-2)."""

from __future__ import annotations

import pytest

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.registry.tool_requirements import (
    SkillToolRequirementError,
    SkillToolRequirementViolation,
    assert_skill_tool_requirements_satisfied,
    resolve_skill_tool_requirements,
)
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _registry_with_skills(*manifests: SkillManifest) -> SkillRegistry:
    registry = SkillRegistry()
    for manifest in manifests:
        registry.register(manifest)
    return registry


def test_resolve_all_requirements_satisfied() -> None:
    registry = _registry_with_skills(
        SkillManifest(skill_id="skill.one", description="one", tool_ids=("tool.a", "tool.c")),
    )
    resolution = resolve_skill_tool_requirements(registry, ("tool.a", "tool.b", "tool.c"))

    assert resolution.missing_tool_ids == ()
    assert resolution.satisfied_tool_ids == ("tool.a", "tool.c")
    assert resolution.is_satisfied is True


def test_resolve_missing_requirement() -> None:
    registry = _registry_with_skills(
        SkillManifest(skill_id="skill.one", description="one", tool_ids=("tool.a", "tool.b")),
    )
    resolution = resolve_skill_tool_requirements(registry, ("tool.a",))

    assert resolution.missing_tool_ids == ("tool.b",)
    assert resolution.is_satisfied is False


def test_resolve_no_requirements_is_satisfied() -> None:
    registry = _registry_with_skills(
        SkillManifest(skill_id="skill.empty", description="empty"),
    )
    resolution = resolve_skill_tool_requirements(registry, ("tool.a",))

    assert resolution.required_tool_ids == ()
    assert resolution.is_satisfied is True


def test_resolve_is_deterministic_for_input_order() -> None:
    manifest_a = SkillManifest(skill_id="skill.a", description="a", tool_ids=("tool.z", "tool.a"))
    manifest_b = SkillManifest(skill_id="skill.b", description="b", tool_ids=("tool.b",))
    registry_forward = _registry_with_skills(manifest_a, manifest_b)
    registry_reverse = _registry_with_skills(manifest_b, manifest_a)

    forward = resolve_skill_tool_requirements(registry_forward, ("tool.a",))
    reverse = resolve_skill_tool_requirements(registry_reverse, ("tool.a",))

    assert forward == reverse


def test_resolve_multiple_skills_union_and_provenance() -> None:
    registry = _registry_with_skills(
        SkillManifest(skill_id="skill.one", description="one", tool_ids=("tool.a", "tool.b")),
        SkillManifest(skill_id="skill.two", description="two", tool_ids=("tool.b", "tool.c")),
    )
    resolution = resolve_skill_tool_requirements(registry, ("tool.a", "tool.b"))

    assert resolution.missing_tool_ids == ("tool.c",)
    assert resolution.violations == (
        SkillToolRequirementViolation(skill_id="skill.two", tool_id="tool.c"),
    )


def test_resolve_custom_plugin_skill_uses_same_failure_semantics() -> None:
    registry = _registry_with_skills(
        SkillManifest(
            skill_id="plugin.custom",
            description="plugin",
            tool_ids=("plugin.tool",),
        ),
    )

    with pytest.raises(SkillToolRequirementError) as exc_info:
        assert_skill_tool_requirements_satisfied(registry, ToolProfile(enabled=["tool.a"]))

    assert exc_info.value.resolution.missing_tool_ids == ("plugin.tool",)
    assert exc_info.value.resolution.violations[0].skill_id == "plugin.custom"


@pytest.fixture
def _isolated_tool_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()

    def register_bundle(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del registry, ctx

    register_tool_bundle(
        ToolBundleEntry(bundle_id="demo", tool_ids=("bundle.tool.a",), register=register_bundle)
    )
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_resolve_bundle_enabled_tool_is_available(_isolated_tool_catalog: None) -> None:
    register_default_tools()
    registry = _registry_with_skills(
        SkillManifest(
            skill_id="skill.bundle",
            description="bundle",
            tool_ids=("bundle.tool.a",),
        ),
    )
    tool_profile = ToolProfile(enabled_bundles=["demo"])

    resolution = assert_skill_tool_requirements_satisfied(registry, tool_profile)

    assert resolution.is_satisfied is True
    assert resolution.satisfied_tool_ids == ("bundle.tool.a",)
