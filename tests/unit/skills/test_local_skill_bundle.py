# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.providers.local.manifests import (
    LOCAL_WORKSPACE_INDEX,
    LOCAL_WORKSPACE_SEARCH,
    LOCAL_WORKSPACE_SYNTHESIZE,
)
from intergrax.skills.providers.local.plugin import LocalSkillPlugin
from intergrax.skills.registry import SkillProfile, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests


pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_skills() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def test_local_bundle_manifest_shape() -> None:
    manifest = LocalSkillPlugin.skill_bundle_manifest()
    assert manifest.bundle_id == "local"
    assert manifest.skill_ids == (
        "local.workspace.index",
        "local.workspace.search",
        "local.workspace.synthesize",
    )


def test_local_bundle_registers_workspace_skills() -> None:
    registry = build_registry_from_profile(SkillProfile(enabled_bundles=["local"]))
    for skill_manifest in (
        LOCAL_WORKSPACE_INDEX,
        LOCAL_WORKSPACE_SEARCH,
        LOCAL_WORKSPACE_SYNTHESIZE,
    ):
        assert registry.has(skill_manifest.skill_id)


def test_local_skill_tool_ids_match_lkw_agents() -> None:
    assert LOCAL_WORKSPACE_INDEX.tool_ids == ("rag.ingest_document",)
    assert LOCAL_WORKSPACE_SEARCH.tool_ids == ("rag.retrieve",)
    assert LOCAL_WORKSPACE_SYNTHESIZE.tool_ids == ("workspace.write_file",)
