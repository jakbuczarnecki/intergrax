# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_workspace_application.host.factory import create_local_workspace_backend_app

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/local_workspace"


def test_local_workspace_backend_health():
    client = TestClient(create_local_workspace_backend_app())
    response = client.get("/health")
    assert response.status_code == 200


def test_local_workspace_backend_lists_agents():
    client = TestClient(create_local_workspace_backend_app())
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_local_workspace_backend_run():
    client = TestClient(create_local_workspace_backend_app())
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "local.workspace.search"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"
