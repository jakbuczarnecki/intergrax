# © Artur Czarnecki. All rights reserved.

"""Runtime MCP mount checks — excluded from CI gate purity (TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from intergrax.utils import attribute_access
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _http_only_settings(**overrides: object) -> LocalWorkspaceBackendSettings:
    base = {
        "environment": LocalWorkspaceBackendSettings.from_env().environment,
        "include_mcp": False,
        "include_scheduler": False,
        "include_task_control": False,
        "include_interaction_routes": False,
    }
    base.update(overrides)
    return LocalWorkspaceBackendSettings(**base)  # type: ignore[arg-type]


def test_mcp_enabled_factory_mounts_mcp_route_when_available() -> None:
    from local_workspace_application.host.factory import create_local_workspace_backend_app

    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(_http_only_settings(include_mcp=True), settings=_http_only_settings(include_mcp=True))
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert any(
        attribute_access.optional(route, "path", None) in {"/mcp", "/mcp/"}
        for route in app.routes
        if hasattr(route, "path")
    )
