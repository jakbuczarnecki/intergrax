# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.lifecycle import HostLifecycleState
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


def test_fastapi_startup_moves_host_to_ready() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    app = create_local_workspace_backend_app(settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    assert lifecycle.state is HostLifecycleState.STARTING
    with TestClient(app):
        assert lifecycle.state is HostLifecycleState.READY
        assert lifecycle.accepts_new_work is True


def test_readiness_endpoint_reports_ready_while_running() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    app = create_local_workspace_backend_app(settings=settings)
    with TestClient(app) as client:
        response = client.get(f"{_PREFIX}/readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["accepts_new_work"] is True
    assert body["state"] == "ready"


def test_health_contract_remains_compatible() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    app = create_local_workspace_backend_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_shutdown_moves_host_to_stopped() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    app = create_local_workspace_backend_app(settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    with TestClient(app):
        assert lifecycle.state is HostLifecycleState.READY
    assert lifecycle.state is HostLifecycleState.STOPPED
