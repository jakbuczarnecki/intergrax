# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from intergrax.utils import attribute_access

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_lab_application_exposes_mcp_mount() -> None:
    from intergrax.fastapi_core.config import ApiEnvironment

    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    settings = LabApplicationSettings(
        environment=ApiEnvironment.DEV,
        include_mcp=True,
        include_scheduler=False,
    )
    app = create_lab_application(settings=settings)
    client = TestClient(app)
    assert client.get("/v1/lab/agents").status_code == 200
    assert any(
        attribute_access.optional(r, "path", None) in {"/mcp", "/mcp/"} for r in app.routes if hasattr(r, "path")
    )


def test_research_application_exposes_mcp_mount(
    configured_research_llm: None,
) -> None:
    """Tier-3 research MCP facade mount (not full strict production factory bootstrap).

    Full ``create_research_backend_app`` under strict PRODUCT still requires revision-bound
    agent contract authority and vendor MemoryProvider qualification (PROD-Q) — see
    ``LIM-RESEARCH-STRICT-FACTORY-MEMORY`` in P5 dossier.
    """
    from intergrax.applications._shared.mcp_host_mount import couple_mcp_fastmcp_on_harness_host
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

    from research_application.mcp.server import build_research_mcp_server

    api = FastAPI()
    registry = AgentRegistry()
    nexus = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(
        nexus,
        orchestration_triggers=frozenset(),
        pipeline_capability_suffix=".pipeline",
    )
    mcp = build_research_mcp_server(
        host_execution=host_execution,
        registry=registry,
        route_prefix="/v1/research",
    )
    combined = couple_mcp_fastmcp_on_harness_host(
        api,
        mcp,
        mount_path="/mcp",
        mcp_enabled=True,
    )
    assert any(
        attribute_access.optional(r, "path", None) in {"/mcp", "/mcp/"} for r in combined.routes if hasattr(r, "path")
    )
