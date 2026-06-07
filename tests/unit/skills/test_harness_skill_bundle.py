# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.harness.manifests import (
    HARNESS_CONTEXT_DEMO,
    HARNESS_INTEGRATION_BRIDGE_SMOKE,
    HARNESS_SKILL_REGISTRY,
    HARNESS_TOOL_SMOKE,
    HARNESS_TRACE_READ,
    HARNESS_MODALITY_SMOKE,
    HARNESS_VISION_QA,
)
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.resolver import SkillResolver


pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_skills() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def test_harness_bundle_registers_platform_skills() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))
    for manifest in (
        HARNESS_TOOL_SMOKE,
        HARNESS_CONTEXT_DEMO,
        HARNESS_TRACE_READ,
        HARNESS_SKILL_REGISTRY,
        HARNESS_MODALITY_SMOKE,
        HARNESS_VISION_QA,
        HARNESS_INTEGRATION_BRIDGE_SMOKE,
    ):
        assert registry.has(manifest.skill_id)


def test_harness_integration_bridge_smoke_merges_bridge_tools() -> None:
    from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
    from intergrax.tools.registry.catalog import clear_tool_catalog
    from intergrax.tools.registry.factory import build_registry_from_profile as build_tool_registry
    from intergrax.tools.registry.profile import ToolProfile

    clear_tool_catalog()
    reset_default_tools_bootstrap()
    register_default_tools()
    tool_registry = build_tool_registry(
        ToolProfile(enabled=["storage.get", "knowledge.search"]),
        ctx=None,
    )
    skill_registry = build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))
    pack = SkillResolver(skill_registry, tool_registry).resolve([HARNESS_INTEGRATION_BRIDGE_SMOKE.skill_id])
    assert pack.tool_ids == frozenset({"storage.get", "knowledge.search"})


def test_harness_skill_resolver_merges_platform_pack() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))
    pack = SkillResolver(registry).resolve(
        [
            HARNESS_TOOL_SMOKE.skill_id,
            HARNESS_TRACE_READ.skill_id,
        ]
    )
    assert "rag.retrieve" in pack.tool_ids
    assert "websearch.query" in pack.tool_ids
    assert "harness.get_run" in pack.tool_ids
    assert "harness.get_run_events" in pack.tool_ids
