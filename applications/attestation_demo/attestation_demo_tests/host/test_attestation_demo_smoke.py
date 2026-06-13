# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from attestation_demo.host.factory import create_attestation_demo_application

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/attestation_demo"


def test_attestation_demo_lists_agents():
    client = TestClient(create_attestation_demo_application())
    response = client.get(f"{_PREFIX}/agents")
    assert response.status_code == 200
    payload = response.json()
    assert "agents" in payload
    agent_ids = [item["agent_id"] for item in payload["agents"]]
    assert "boundary_demo_agent" in agent_ids


def test_attestation_demo_poc_run_returns_boundary_events():
    client = TestClient(create_attestation_demo_application())
    response = client.post(
        f"{_PREFIX}/poc/run",
        json={
            "message": "Partner PoC sample",
            "capability": "attestation.demo",
            "partition_key": "attestation_demo",
            "row_key": "poc-smoke-001",
            "record_data": {"title": "Partner PoC sample", "version": 1},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("state") == "completed"
    assert body.get("agent_id") == "boundary_demo_agent"
    events = body.get("boundary_events") or []
    assert len(events) >= 1
    event = events[0]
    assert event.get("schema_id") == "execution_boundary_event.v1"
    assert event.get("signed") is False
    assert event.get("tool_id") == "records.put"
    assert event.get("agent_id") == "boundary_demo_agent"
    assert event.get("action_status") == "executed"
    assert event.get("input", {}).get("partition_key") == "attestation_demo"
    assert body.get("trust_model", {}).get("recommended_receipt_role") == "client_observed"

    run_id = body.get("run_id")
    assert run_id
    debug = client.get(f"{_PREFIX}/poc/runs/{run_id}/boundary-events")
    assert debug.status_code == 200
    assert debug.json().get("count") == len(events)
