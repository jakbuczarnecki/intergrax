# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = pytest.mark.unit

_PREFIX = "/v1/local_workspace"
_INTAKE_PREFIX = "/v1/interactions"


def _settings_with_interactions() -> LocalWorkspaceBackendSettings:
    return LocalWorkspaceBackendSettings(
        include_interaction_routes=True,
        include_mcp=False,
        include_scheduler=False,
        interaction_surface="lab_json",
        interaction_execute_default=True,
    )


def test_run_and_interaction_intake_share_task_executor() -> None:
    settings = _settings_with_interactions()
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    shared_executor = app.state.lkw_task_executor
    with TestClient(app) as client:
        run_response = client.post(
            f"{_PREFIX}/run",
            json={"message": "hello", "capability": "local.workspace.search"},
        )
        intake_response = client.post(
            f"{_INTAKE_PREFIX}/intake?execute=true&tenant=t1",
            json={
                "message": "hello",
                "capability": "local.workspace.search",
                "user_id": "u1",
            },
        )
    assert run_response.status_code == 200
    assert intake_response.status_code == 200
    assert app.state.lkw_task_executor is shared_executor


def test_interaction_metadata_survives_execution() -> None:
    settings = _settings_with_interactions()
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    with TestClient(app) as client:
        response = client.post(
            f"{_INTAKE_PREFIX}/intake?execute=false&tenant=t1",
            json={
                "message": "hello",
                "capability": "local.workspace.search",
                "user_id": "u1",
                "metadata": {"proof_marker": "lkw-6a"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "local.workspace.search"
    assert body["interaction_channel"] == "lab"


def test_run_and_intake_reject_work_when_not_ready() -> None:
    settings = _settings_with_interactions()
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection(settings), settings=settings)
    lifecycle = app.state.lkw_host_lifecycle
    with TestClient(app) as client:
        lifecycle.transition_to_stopping()
        run_response = client.post(
            f"{_PREFIX}/run",
            json={"message": "hello", "capability": "local.workspace.search"},
        )
        intake_response = client.post(
            f"{_INTAKE_PREFIX}/intake?execute=true&tenant=t1",
            json={
                "message": "hello",
                "capability": "local.workspace.search",
                "user_id": "u1",
            },
        )
    assert run_response.status_code == 503
    assert run_response.json()["detail"]["error_id"] == "lkw_host_stopping"
    assert intake_response.status_code == 503
    assert intake_response.json()["detail"]["error_id"] == "lkw_host_stopping"
