# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.tests.dispute_sim_ac3_projection import (
    build_dispute_sim_test_registry_projection,
)

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/dispute_sim"


def test_dispute_sim_backend_health():
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.get("/health")
    assert response.status_code == 200


def test_dispute_sim_backend_lists_agents():
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    assert "agents" in response.json()


def test_dispute_sim_backend_run():
    client = TestClient(
        create_dispute_sim_backend_app(
            registry_projection=build_dispute_sim_test_registry_projection(),
        )
    )
    response = client.post(
        f"{_PREFIX}/run",
        json={"message": "hello", "capability": "dispute.intake"},
    )
    assert response.status_code == 200
    assert response.json().get("state") == "completed"
