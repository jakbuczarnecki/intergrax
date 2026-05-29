# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from lab_application.host.factory import create_lab_application

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]


@pytest.fixture
def lab_client(tmp_path):
    app = create_lab_application(
        checkpoints_db_path=tmp_path / "lab.db",
        runtime_events_db_path=tmp_path / "events.db",
        experiments_db_path=tmp_path / "experiments.db",
    )
    return TestClient(app)


def test_lab_application_lists_agents(lab_client: TestClient):
    response = lab_client.get("/v1/lab/agents")
    assert response.status_code == 200
    agents = response.json()["agents"]
    agent_ids = {item["agent_id"] for item in agents}
    assert "echo" in agent_ids
    assert "research_mock" in agent_ids


def test_lab_application_runs_echo_agent(lab_client: TestClient):
    response = lab_client.post(
        "/v1/lab/run",
        json={
            "tenant_id": "lab",
            "user_id": "tester",
            "message": "lab acceptance",
            "capability": "echo.basic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "completed"
    assert "lab acceptance" in body["answer"]
    assert body["agent_id"] == "echo"
