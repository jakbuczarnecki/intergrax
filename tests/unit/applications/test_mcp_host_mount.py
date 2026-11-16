# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.mcp_host_mount import (
    McpHostMountError,
    compose_mcp_fastmcp_for_definition,
    couple_mcp_fastmcp_on_harness_host,
    normalize_mcp_mount_path,
    resolve_mcp_mount_path_for_host,
)
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    nexus_legacy_server_definition,
)
from intergrax.contracts.mcp_server_definition import McpExposurePolicy, build_mcp_server_definition
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

pytestmark = [pytest.mark.unit]


def test_normalize_mcp_mount_path_rejects_api_root() -> None:
    with pytest.raises(McpHostMountError):
        normalize_mcp_mount_path("/")


def test_resolve_mount_passthrough_when_disabled() -> None:
    assert resolve_mcp_mount_path_for_host("bad", enabled=False) == "bad"


def test_compose_invalid_definition_fails_at_startup() -> None:
    definition = build_mcp_server_definition(
        server_id="x",
        display_name="X",
        default_capability="cap",
        default_tenant_id="t",
        default_user_id="u",
        bindings=(),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("missing",)),
    )
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    deps = McpServerCompositionDeps(
        host_execution=build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset()),
        registry=registry,
    )
    with pytest.raises(Exception):
        compose_mcp_fastmcp_for_definition(definition, deps)


def test_couple_engine_mount_serves_mcp_and_api() -> None:

    from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server

    api = FastAPI()

    @api.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "1"}

    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    mcp = build_nexus_mcp_server(
        name="MountTest",
        host_execution=host_execution,
        registry=registry,
        default_capability="test",
    )
    combined = couple_mcp_fastmcp_on_harness_host(api, mcp, mount_path="/mcp", mcp_enabled=True)
    client = TestClient(combined)
    assert client.get("/ping").json() == {"ok": "1"}
    assert any(getattr(r, "path", None) == "/mcp" for r in combined.routes)


def test_compose_legacy_definition_mount_roundtrip() -> None:
    api = FastAPI()
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    definition = nexus_legacy_server_definition(
        server_id="demo",
        display_name="Demo",
        default_capability="c",
        default_tenant_id="t",
        default_user_id="u",
        catalog_tool_ids=(),
        include_catalog_surface=False,
    )
    deps = McpServerCompositionDeps(
        host_execution=build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset()),
        registry=registry,
    )
    mcp = compose_mcp_fastmcp_for_definition(definition, deps)
    combined = couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")
    assert TestClient(combined).get("/docs").status_code in {200, 404}
