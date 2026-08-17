# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.tests.lkw_ac3_projection import build_lkw_test_registry_projection

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/local_workspace"


def test_local_workspace_backend_health():
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection())
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_local_workspace_backend_lists_agents():
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection())
    with TestClient(app) as client:
        response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_local_workspace_backend_run():
    app = create_local_workspace_backend_app(registry_projection=build_lkw_test_registry_projection())
    with TestClient(app) as client:
        response = client.post(
            f"{_PREFIX}/run",
            json={"message": "hello", "capability": "local.workspace.search"},
        )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"
