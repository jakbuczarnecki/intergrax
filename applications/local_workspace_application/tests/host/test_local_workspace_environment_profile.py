# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.skill_wiring import build_application_skill_wiring
from intergrax.skills.integration.contract_resolution import resolve_contract_tools
from intergrax.skills.registry.bootstrap import register_default_skills, reset_default_skills_for_tests
from intergrax.skills.resolver import SkillResolver
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_workspace_application.host.environment_profile import build_local_workspace_environment_profile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_skills() -> None:
    reset_default_skills_for_tests()
    register_default_skills()
    yield
    reset_default_skills_for_tests()


def test_lkw_environment_profile_enables_harness_and_local_bundles() -> None:
    env = build_local_workspace_environment_profile()
    assert "harness" in env.skill_profile.enabled_bundles
    assert "local" in env.skill_profile.enabled_bundles


def test_lkw_environment_profile_registers_local_workspace_skills() -> None:
    env = build_local_workspace_environment_profile()
    wiring = build_application_skill_wiring(env.skill_profile)
    for skill_id in (
        "local.workspace.index",
        "local.workspace.search",
        "local.workspace.synthesize",
    ):
        assert wiring.registry.has(skill_id)


def test_lkw_environment_resolves_local_indexer_tools() -> None:
    env = build_local_workspace_environment_profile()
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    contract = LocalIndexerAgent().get_contract()
    resolver = SkillResolver(skill_wiring.registry, tool_registry=None)
    resolved, _ = resolve_contract_tools(contract, skill_resolver=resolver)
    assert RAG_INGEST_TOOL_ID in resolved.allowed_tools
