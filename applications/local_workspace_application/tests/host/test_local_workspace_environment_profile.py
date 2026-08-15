# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)
from intergrax.applications._shared.package_wiring import _validate_graph_spec_capabilities
from intergrax.applications._shared.package_wiring import (
    build_application_package,
    validate_application_package_closure,
)
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
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
    build_local_workspace_integration_profile,
)
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP_ROOT = Path(__file__).resolve().parents[2]
_DOCKER_COMPOSE = _APP_ROOT / "docker" / "docker-compose.yml"


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


def test_lkw_integration_profile_defaults_to_qdrant_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_WORKSPACE_VECTOR_STORE", raising=False)
    monkeypatch.delenv("LOCAL_WORKSPACE_ENABLE_REDIS", raising=False)

    profile = build_local_workspace_integration_profile()

    assert profile.vector_store is not None
    assert profile.vector_store.manifest.slug == "qdrant"
    assert "qdrant" in profile.options


def test_lkw_integration_profile_allows_explicit_inmemory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")

    profile = build_local_workspace_integration_profile()

    assert profile.vector_store is not None
    assert profile.vector_store.manifest.slug == "inmemory"
    assert "inmemory" in profile.options


def test_lkw_integration_profile_rejects_unsupported_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "chroma")

    with pytest.raises(
        ValueError,
        match="LOCAL_WORKSPACE_VECTOR_STORE must be one of: qdrant, inmemory",
    ):
        build_local_workspace_integration_profile()


def test_lkw_docker_compose_uses_persistent_qdrant_storage_contract() -> None:
    text = _DOCKER_COMPOSE.read_text(encoding="utf-8")

    assert "LOCAL_WORKSPACE_VECTOR_STORE: qdrant" in text
    assert "INTERGRAX_QDRANT_URL: http://qdrant:6333" in text
    assert "INTERGRAX_QDRANT_COLLECTION: local_workspace" in text
    assert "\n  qdrant:" in text
    assert "qdrant_data:/qdrant/storage" in text
    assert "\n  qdrant_data:" in text


def test_lkw_docker_compose_uses_persistent_otel_collector_storage_contract() -> None:
    text = _DOCKER_COMPOSE.read_text(encoding="utf-8")

    assert "\n  otel-collector:" in text
    assert (
        "./otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml:ro" in text
    )
    assert "lkw_otel_data:/var/lib/otelcol" in text
    assert "\n  lkw_otel_data:" in text
    assert '"4318:4318"' in text


def test_lkw_package_closure_accepts_pipeline_graph_trigger() -> None:
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    env = manifest.resolved_environment()
    wiring = wire_application_environment(manifest, env, conformance_check=False)
    package = build_application_package(manifest, env)
    violations = validate_application_package_closure(
        package,
        manifest,
        env,
        wiring.registry_snapshot,
        capability_graph=wiring.capability_graph,
    )
    roster_violations = [
        v for v in violations if "not found on manifest roster" in v or "graph trigger capability" in v
    ]
    assert roster_violations == [], f"unexpected roster/graph violations: {violations}"
    roster_caps = {
        capability
        for binding in manifest.enabled_agents()
        for capability in binding.capabilities
    }
    assert "local.workspace.pipeline" not in roster_caps


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
    assert {node.contract_id for node in spec.nodes} == {
        "LocalIndexerAgent",
        "LocalSearchAgent",
        "LocalSynthesizerAgent",
    }
    edges = [(edge.source_agent_id, edge.target_agent_id) for edge in spec.edges]
    assert edges == [
        ("local_indexer", "local_search"),
        ("local_search", "local_synthesizer"),
    ]


def test_lkw_graph_spec_validates_against_manifest_roster() -> None:
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    env = manifest.resolved_environment()
    spec = env.graph_spec
    assert spec is not None
    spec.validate_against_roster(manifest.enabled_agents())
    assert _validate_graph_spec_capabilities(manifest, env) == []


def test_lkw_graph_plan_preserves_routing_agent_ids() -> None:
    env = build_local_workspace_environment_profile()
    spec = env.graph_spec
    assert spec is not None
    task = Task(
        tenant_id="tenant-lkw",
        user_id="user-lkw",
        message="pipeline",
        context=TaskContext(capability="local.workspace.pipeline"),
    )
    plan = application_graph_spec_to_nexus_plan(spec, task, classification="multi_agent_default")
    assert [step.agent_id for step in plan.steps] == [
        "local_indexer",
        "local_search",
        "local_synthesizer",
    ]


def test_lkw_invalid_graph_node_fails_package_closure() -> None:
    manifest = LOCAL_WORKSPACE_APPLICATION_MANIFEST
    env = manifest.resolved_environment()
    assert env.graph_spec is not None
    invalid_node = env.graph_spec.nodes[0].model_copy(
        update={"agent_id": "missing_agent", "contract_id": None},
    )
    invalid_spec = env.graph_spec.model_copy(
        update={"nodes": list(env.graph_spec.nodes) + [invalid_node]},
    )
    invalid_env = env.model_copy(update={"graph_spec": invalid_spec})
    violations = _validate_graph_spec_capabilities(manifest, invalid_env)
    assert any("not found on manifest roster" in violation for violation in violations)


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
