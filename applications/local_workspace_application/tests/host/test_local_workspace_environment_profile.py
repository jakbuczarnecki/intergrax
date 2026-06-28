# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.graph_spec_to_plan import should_seed_plan_from_graph_spec
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring
from intergrax.runtime.nexus.orchestration_capabilities import (
    is_orchestration_capability,
    orchestration_capabilities_from_triggers,
)
from intergrax.runtime.task.task import Task, TaskContext
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


def test_lkw_environment_profile_registers_pipeline_graph_spec() -> None:
    env = build_local_workspace_environment_profile()
    spec = env.graph_spec
    assert spec is not None
    assert spec.trigger_capabilities == ["local.workspace.pipeline"]
    assert {node.agent_id for node in spec.nodes} == {
        "local_indexer",
        "local_search",
        "local_synthesizer",
    }
    edges = [(edge.source_agent_id, edge.target_agent_id) for edge in spec.edges]
    assert edges == [
        ("local_indexer", "local_search"),
        ("local_search", "local_synthesizer"),
    ]


def test_lkw_pipeline_is_orchestration_trigger_capability() -> None:
    env = build_local_workspace_environment_profile()
    spec = env.graph_spec
    assert spec is not None
    triggers = orchestration_capabilities_from_triggers(spec.trigger_capabilities)
    assert is_orchestration_capability(
        "local.workspace.pipeline",
        trigger_capabilities=triggers,
        pipeline_capability_suffix=spec.pipeline_capability_suffix,
    )
    task = Task(
        tenant_id="tenant-lkw",
        user_id="user-lkw",
        message="pipeline",
        context=TaskContext(capability="local.workspace.pipeline"),
    )
    assert should_seed_plan_from_graph_spec(task, spec) is True


def test_lkw_environment_resolves_local_indexer_tools() -> None:
    env = build_local_workspace_environment_profile()
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    contract = LocalIndexerAgent().get_contract()
    resolver = SkillResolver(skill_wiring.registry, tool_registry=None)
    resolved, _ = resolve_contract_tools(contract, skill_resolver=resolver)
    assert RAG_INGEST_TOOL_ID in resolved.allowed_tools
