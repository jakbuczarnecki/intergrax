# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from poc_template_application.host.factory import create_poc_template_application

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/poc_template"


def test_poc_template_application_lists_agents():
    client = TestClient(create_poc_template_application())
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    payload = response.json()
    assert "agents" in payload
    assert len(payload["agents"]) >= 1


def test_poc_template_application_run_echo():
    client = TestClient(create_poc_template_application())
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "echo.basic"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("state") == "completed"
