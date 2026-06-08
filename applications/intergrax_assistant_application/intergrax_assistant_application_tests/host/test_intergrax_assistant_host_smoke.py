# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intergrax_assistant_application.host.factory import create_intergrax_assistant_application

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/intergrax_assistant"


def test_intergrax_assistant_application_lists_agents():
    client = TestClient(create_intergrax_assistant_application())
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    payload = response.json()
    assert "agents" in payload
    assert len(payload["agents"]) >= 1


def test_intergrax_assistant_application_run_echo():
    client = TestClient(create_intergrax_assistant_application())
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "platform.assist"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("state") == "completed"
