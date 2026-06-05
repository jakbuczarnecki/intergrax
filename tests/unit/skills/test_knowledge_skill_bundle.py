# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.knowledge.manifests import KNOWLEDGE_OPENAI_STRICT
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


def test_knowledge_bundle_registers_openai_strict_skill() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["knowledge"]))
    assert registry.has(KNOWLEDGE_OPENAI_STRICT.skill_id)


def test_knowledge_openai_strict_resolves_file_search_tool() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["knowledge"]))
    pack = SkillResolver(registry).resolve([KNOWLEDGE_OPENAI_STRICT.skill_id])
    assert "openai.file_search.query" in pack.tool_ids
    assert "knowledge.openai_strict.system" in pack.prompt_instruction_ids
