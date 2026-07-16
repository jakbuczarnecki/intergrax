# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.lifecycle import HostLifecycleState
from local_workspace_application.host.readiness import LocalWorkspaceReadinessSnapshot
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"


@dataclass
class _FakeReadiness:
    snapshot: LocalWorkspaceReadinessSnapshot
    calls: list[LocalWorkspaceReadinessSnapshot] = field(default_factory=list)

    def readiness_snapshot(self) -> LocalWorkspaceReadinessSnapshot:
        self.calls.append(self.snapshot)
        return self.snapshot


def test_fastapi_startup_moves_host_to_ready() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    app = create_local_workspace_backend_app(settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    assert app.state.lkw_host_readiness is lifecycle
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
    assert "detail" in body
    assert "components" in body
    assert "rejection_error_id" not in body


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


def test_hosted_mode_uses_injected_readiness_without_local_lifecycle() -> None:
    settings = LocalWorkspaceBackendSettings(include_mcp=False, include_scheduler=False)
    fake = _FakeReadiness(
        snapshot=LocalWorkspaceReadinessSnapshot(
            ready=False,
            accepts_new_work=False,
            state="starting",
            detail="host_state=starting",
            rejection_error_id="lkw_host_not_ready",
        )
    )
    app = create_local_workspace_backend_app(
        settings=settings,
        host_readiness=fake,
    )
    assert app.state.lkw_host_readiness is fake
    assert not hasattr(app.state, "lkw_host_lifecycle")

    with TestClient(app) as client:
        starting = client.get(f"{_PREFIX}/readiness")
        assert starting.status_code == 200
        assert starting.json()["state"] == "starting"
        assert starting.json()["accepts_new_work"] is False

        fake.snapshot = LocalWorkspaceReadinessSnapshot(
            ready=True,
            accepts_new_work=True,
            state="ready",
            detail="ready",
            rejection_error_id="",
        )
        ready = client.get(f"{_PREFIX}/readiness")
        assert ready.status_code == 200
        assert ready.json()["state"] == "ready"
        assert ready.json()["ready"] is True
        assert ready.json()["accepts_new_work"] is True

        fake.snapshot = LocalWorkspaceReadinessSnapshot(
            ready=False,
            accepts_new_work=False,
            state="stopping",
            detail="host_state=stopping",
            rejection_error_id="lkw_host_stopping",
        )
        run_response = client.post(
            f"{_PREFIX}/run",
            json={"message": "hello", "capability": "local.workspace.search"},
        )
        assert run_response.status_code == 503
        assert run_response.json()["detail"]["error_id"] == "lkw_host_stopping"

    assert len(fake.calls) >= 3
