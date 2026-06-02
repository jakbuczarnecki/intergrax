# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.harness.manifests import (
    HARNESS_CONTEXT_DEMO,
    HARNESS_SKILL_REGISTRY,
    HARNESS_TOOL_SMOKE,
    HARNESS_TRACE_READ,
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
    ):
        assert registry.has(manifest.skill_id)


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
    assert "sandbox.exec" in pack.tool_ids
