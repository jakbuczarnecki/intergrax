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
    assert len(events) >= 2
    tool_event = next(e for e in events if e.get("boundary_type") == "tool_execution")
    assert tool_event.get("schema_id") == "execution_boundary_event.v1"
    assert tool_event.get("signed") is True
    assert tool_event.get("host_attestation") is not None
    assert tool_event.get("tool_id") == "records.put"
    assert tool_event.get("agent_id") == "boundary_demo_agent"
    assert tool_event.get("action_status") == "executed"
    assert tool_event.get("event_sequence") == 1
    assert tool_event.get("input", {}).get("partition_key") == "attestation_demo"
    assert body.get("trust_model", {}).get("recommended_receipt_role") == "host_attested"

    run_id = body.get("run_id")
    assert run_id
    debug = client.get(f"{_PREFIX}/poc/runs/{run_id}/boundary-events")
    assert debug.status_code == 200
    assert debug.json().get("count") == len(events)


def test_attestation_demo_poc_run_full_boundary_event_contract():
    """Partner handoff — validate execution_boundary_event.v1 fields and debug trace."""
    client = TestClient(create_attestation_demo_application())
    response = client.post(
        f"{_PREFIX}/poc/run",
        json={
            "tenant_id": "default",
            "message": "Partner PoC sample",
            "capability": "attestation.demo",
            "partition_key": "attestation_demo",
            "row_key": "poc-contract-001",
            "record_data": {"title": "PoC report", "version": 1},
        },
    )
    assert response.status_code == 200
    body = response.json()
    tool_event = next(
        event
        for event in body["boundary_events"]
        if event.get("boundary_type") == "tool_execution"
    )

    assert tool_event["schema_id"] == "execution_boundary_event.v1"
    assert tool_event["boundary_type"] == "tool_execution"
    assert tool_event["event_sequence"] == 1
    assert tool_event["signed"] is True
    assert tool_event["host_attestation"] is not None
    assert tool_event["side_effects"] is True
    assert tool_event["risk_level"] == "medium"
    assert tool_event["step_id"] == "store_demo_record"
    assert tool_event["lineage"]["type"] == "execution_record"
    assert tool_event["lineage"]["ref"].startswith(body["run_id"])
    assert tool_event["runtime_ref"]["platform"] == "intergrax"
    assert tool_event["input_hash"] and tool_event["input_hash"].startswith("sha256:")
    assert tool_event["output_hash"] and tool_event["output_hash"].startswith("sha256:")
    harness_event = next(
        event
        for event in body["boundary_events"]
        if event.get("boundary_type") == "harness_step"
    )
    assert harness_event["event_sequence"] == 2
    assert harness_event["action_status"] == "completed"

    trace = client.get(f"/debug/tasks/{body['run_id']}/trace")
    assert trace.status_code == 200
    trace_payload = trace.json()
    assert trace_payload.get("run_id") == body["run_id"]
    trace_events = trace_payload.get("trace_events") or []
    assert len(trace_events) >= 1
    trace_blob = str(trace_events)
    assert (
        "records.put" in trace_blob
        or "store_demo_record" in trace_blob
        or "boundary_demo_agent" in trace_blob
    )


def test_attestation_demo_poc_run_rejects_missing_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "partner-secret")
    client = TestClient(create_attestation_demo_application())
    response = client.post(
        f"{_PREFIX}/poc/run",
        json={"message": "auth check", "capability": "attestation.demo"},
    )
    assert response.status_code == 401

    authorized = client.post(
        f"{_PREFIX}/poc/run",
        headers={"X-Api-Key": "partner-secret"},
        json={"message": "auth check", "capability": "attestation.demo"},
    )
    assert authorized.status_code == 200
