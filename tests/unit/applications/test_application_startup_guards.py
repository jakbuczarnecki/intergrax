# © Artur Czarnecki. All rights reserved.

"""Regression guards for Tier-3 application startup closure (LKW platform lessons)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.applications._shared.capability_graph_wiring import resolve_environment_capability_graph
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.architecture.capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]
SHARED = REPO / "intergrax" / "applications" / "_shared"


def test_resolve_environment_capability_graph_default_skips_global_catalog() -> None:
    manifest = ApplicationManifest.lab(
        app_id="startup_guard_test",
        name="Startup Guard Test",
        route_prefix="/v1/startup_guard_test",
        env_prefix="STARTUP_GUARD_TEST_",
        agents=[],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="startup_guard_test.lab")
    snapshot = HarnessRegistrySnapshot(
        integration_profile=IntegrationProfile(),
        tool_registry=None,
        skill_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )

    with patch(
        "intergrax.runtime.architecture.capability_graph.build_catalog_capability_graph",
        side_effect=AssertionError("default runtime must not load global catalog"),
    ):
        view = resolve_environment_capability_graph(manifest, env, snapshot, catalog=None)

    assert view.contains_node("application:startup_guard_test_application")


def test_resolve_environment_capability_graph_honors_explicit_catalog() -> None:
    manifest = ApplicationManifest.lab(
        app_id="catalog_explicit_test",
        name="Catalog Explicit Test",
        route_prefix="/v1/catalog_explicit_test",
        env_prefix="CATALOG_EXPLICIT_TEST_",
        agents=[],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="catalog_explicit_test.lab")
    snapshot = HarnessRegistrySnapshot(
        integration_profile=IntegrationProfile(),
        tool_registry=None,
        skill_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )
    catalog = CapabilityGraph(
        nodes=[
            CapabilityNode(
                node_id="application:catalog_explicit_test_application",
                node_type=CapabilityNodeType.APPLICATION,
            ),
            CapabilityNode(
                node_id="policy:runtime_policy_bundle",
                node_type=CapabilityNodeType.POLICY,
            ),
        ],
        edges=[],
    )

    view = resolve_environment_capability_graph(manifest, env, snapshot, catalog=catalog)
    assert view.contains_node("application:catalog_explicit_test_application")
    assert view.contains_node("policy:runtime_policy_bundle")


@pytest.mark.parametrize(
    "module_name",
    [
        "intergrax.applications._shared.workspace_cleanup_wiring",
        "intergrax.applications._shared.plugin_bootstrap",
    ],
)
def test_shared_wiring_modules_do_not_import_fastapi_mcp(module_name: str) -> None:
    source_path = SHARED / f"{module_name.rsplit('.', 1)[-1]}.py"
    text = source_path.read_text(encoding="utf-8")
    assert "fastapi_mcp" not in text


def test_lkw_dockerfile_uses_minimal_agent_closure() -> None:
    dockerfile = REPO / "applications" / "local_workspace_application" / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    assert "materialized runtime-graph context" in text
    assert "COPY agents/ ./agents/" in text
    assert "COPY agents/local_indexer/" not in text
    assert "uv sync --frozen --no-dev --project applications/local_workspace_application" in text
    pyproject = (
        REPO / "applications" / "local_workspace_application" / "pyproject.toml"
    ).read_text(encoding="utf-8")
    for dist in (
        "intergrax-local-indexer-agent",
        "intergrax-local-search-agent",
        "intergrax-local-synthesizer-agent",
    ):
        assert dist in pyproject


@pytest.mark.no_ci
def test_local_workspace_factory_http_only_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_INTERACTIONS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")

    from local_workspace_application.host.factory import create_local_workspace_backend_app

    app = create_local_workspace_backend_app()
    assert app.title
    assert "Local Workspace" in app.title


def test_scaffold_dockerfile_has_build_smoke_and_compose_project_name(tmp_path: Path) -> None:
    from intergrax.scaffold.new_application import create_application

    target = create_application(
        name="docker_guard_test",
        agents=["echo"],
        profile="product",
        root=tmp_path,
        port=8299,
        force=True,
    )
    dockerfile = (target / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compose = (target / "docker" / "docker-compose.yml").read_text(encoding="utf-8")

    assert "COPY agents/ ./agents/" in dockerfile
    assert "COPY agents/echo/" not in dockerfile
    assert "materialized runtime-graph context" in dockerfile
    assert "create_docker_guard_test_backend_app()" in dockerfile
    assert "name: intergrax_docker_guard_test" in compose
    assert "context: ./runtime-context" in compose
    pyproject = (target / "pyproject.toml").read_text(encoding="utf-8")
    assert "intergrax-echo-agent" in pyproject
