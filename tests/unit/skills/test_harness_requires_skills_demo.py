# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.harness.manifests import HARNESS_STACK_DEMO, HARNESS_TOOL_SMOKE
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.resolver import SkillResolver

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def test_harness_stack_demo_merges_dependency_tools() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["harness"]))
    pack = SkillResolver(registry).resolve([HARNESS_STACK_DEMO.skill_id])
    assert HARNESS_TOOL_SMOKE.skill_id in pack.skill_ids
    assert "websearch.read_url" in pack.tool_ids
    assert "rag.retrieve" in pack.tool_ids
