# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from lab_application.host.factory import create_lab_application

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]


@pytest.fixture
def lab_client(tmp_path):
    app = create_lab_application(
        db_path=tmp_path / "trace.db",
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
    assert "signoff_probe" in agent_ids


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


def test_lab_application_runs_signoff_probe_with_trace(lab_client: TestClient):
    run_response = lab_client.post(
        "/v1/lab/run",
        json={
            "tenant_id": "lab",
            "user_id": "tester",
            "message": "signoff observability proof",
            "capability": "signoff.probe",
        },
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["state"] == "completed"
    assert run_body["agent_id"] == "signoff_probe"
    assert "signoff observability proof" in run_body["answer"]

    task_id = run_body["task_id"]
    tenant = "lab"

    detail_response = lab_client.get(f"/debug/tasks/{task_id}", params={"tenant": tenant})
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["run_id"] == task_id
    assert detail["tenant_id"] == tenant
    assert detail["event_count"] > 0

    trace_response = lab_client.get(
        f"/debug/tasks/{task_id}/trace",
        params={"tenant": tenant, "include_runtime": "true"},
    )
    assert trace_response.status_code == 200
    trace = trace_response.json()
    assert trace["run_id"] == task_id
    assert len(trace["trace_events"]) > 0
    assert trace["runtime_events"]
    assert len(trace["runtime_events"]) > 0

    events_response = lab_client.get(
        f"/debug/tasks/{task_id}/events",
        params={"tenant": tenant},
    )
    assert events_response.status_code == 200
    events = events_response.json()
    assert events["count"] > 0


def test_lab_application_interaction_intake_json_only(lab_client: TestClient):
    response = lab_client.post(
        "/v1/interactions/intake",
        params={"tenant": "lab", "execute": "false"},
        json={
            "command": "/intergrax",
            "text": "echo.basic hello prod route",
            "user_id": "U1",
            "team_id": "T1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "echo.basic"
    assert body["message"] == "hello prod route"
    assert body["interaction_channel"] == "slash_command"
    assert body["executed"] is False


def test_lab_application_interaction_intake_execute(lab_client: TestClient):
    response = lab_client.post(
        "/v1/interactions/intake",
        params={"tenant": "lab", "execute": "true"},
        json={
            "command": "/intergrax",
            "text": "echo.basic run via prod intake",
            "user_id": "U1",
            "team_id": "T1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is True
    assert body["state"] == "completed"
    assert "run via prod intake" in (body["answer"] or "")
