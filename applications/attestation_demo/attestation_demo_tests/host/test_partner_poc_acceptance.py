# © Artur Czarnecki. All rights reserved.

"""Acceptance tests aligned with AgentReceipt partner PoC agreement (v2).

Validates:
- execution-boundary events delivered in trigger API response (no webhook)
- tool_execution + harness_step events with event_sequence ordering
- unsigned events + honest trust_model (client_observed, not server attested)
- records.put tool boundary + HarnessKernel step boundary
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from attestation_demo.host.factory import create_attestation_demo_application
from attestation_demo.partner_handoff.contract_assertions import (
    assert_partner_harness_boundary_event,
    assert_partner_poc_response_shape,
    assert_partner_tool_boundary_event,
)

_PREFIX = "/v1/attestation_demo"
_HANDOFF_DIR = Path(__file__).resolve().parents[2] / "partner_handoff"
_SAMPLE_REQUEST = _HANDOFF_DIR / "poc_run_request.v1.json"

pytestmark = [pytest.mark.unit]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_attestation_demo_application())


@pytest.fixture
def poc_response(client: TestClient) -> dict:
    body = json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8"))
    response = client.post(f"{_PREFIX}/poc/run", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _tool_event(poc_response: dict) -> dict:
    return next(
        event
        for event in poc_response["boundary_events"]
        if event.get("boundary_type") == "tool_execution"
    )


def _harness_event(poc_response: dict) -> dict:
    return next(
        event
        for event in poc_response["boundary_events"]
        if event.get("boundary_type") == "harness_step"
    )


def test_partner_delivery_via_trigger_api_response_not_webhook(poc_response: dict) -> None:
    assert_partner_poc_response_shape(poc_response)
    assert poc_response.get("state") == "completed"


def test_partner_v2_event_sequence_and_types(poc_response: dict) -> None:
    events = poc_response["boundary_events"]
    assert [event["event_sequence"] for event in events] == [1, 2]
    assert {event["boundary_type"] for event in events} == {
        "tool_execution",
        "harness_step",
    }


def test_partner_agent_and_tool_identity(poc_response: dict) -> None:
    tool_event = _tool_event(poc_response)
    harness_event = _harness_event(poc_response)
    assert poc_response.get("agent_id") == "boundary_demo_agent"
    assert tool_event["agent_id"] == "boundary_demo_agent"
    assert harness_event["agent_id"] == "boundary_demo_agent"
    assert tool_event["tool_id"] == "records.put"


def test_partner_executed_status_and_canonical_io(poc_response: dict) -> None:
    tool_event = _tool_event(poc_response)
    assert tool_event["action_status"] == "executed"
    assert isinstance(tool_event["input"], dict)
    assert isinstance(tool_event["output"], dict)
    assert tool_event["input"].get("partition_key") == "attestation_demo"
    assert tool_event["input"].get("data", {}).get("title") == "PoC report"
    assert tool_event["output"].get("stored") is True


def test_partner_harness_step_completed(poc_response: dict) -> None:
    harness_event = _harness_event(poc_response)
    assert harness_event["action_status"] == "completed"
    assert harness_event["step_outcome"]["status"] == "completed"
    assert harness_event["step_outcome"]["outcome_applied"] is True
    assert harness_event["policy_verdicts"]


def test_partner_optional_hashes_for_cross_check(poc_response: dict) -> None:
    for event in poc_response["boundary_events"]:
        assert str(event.get("input_hash", "")).startswith("sha256:")
        assert str(event.get("output_hash", "")).startswith("sha256:")


def test_partner_run_id_step_id_lineage_and_timestamp(poc_response: dict) -> None:
    run_id = poc_response["run_id"]
    tool_event = _tool_event(poc_response)
    harness_event = _harness_event(poc_response)
    assert run_id
    assert tool_event["run_id"] == run_id
    assert harness_event["run_id"] == run_id
    assert tool_event["step_id"] == "store_demo_record"
    assert harness_event["step_id"] == "store_demo_record"
    assert tool_event["lineage"]["type"] == "execution_record"
    assert run_id in tool_event["lineage"]["ref"]
    assert run_id in harness_event["lineage"]["ref"]
    assert tool_event.get("occurred_at")
    assert harness_event.get("occurred_at")


def test_partner_unsigned_event_no_platform_attestation(poc_response: dict) -> None:
    for event in poc_response["boundary_events"]:
        assert event["signed"] is False
    trust = poc_response.get("trust_model") or {}
    assert trust.get("platform_signed") == "false"
    assert trust.get("recommended_receipt_role") == "client_observed"
    note = (trust.get("note") or "").lower()
    assert "unsigned" in note or "partner signs" in note
    assert "server_attested" not in json.dumps(trust).lower()


def test_partner_tool_and_harness_boundaries(poc_response: dict) -> None:
    tool_event = assert_partner_tool_boundary_event(_tool_event(poc_response))
    harness_event = assert_partner_harness_boundary_event(_harness_event(poc_response))
    assert tool_event.boundary_type == "tool_execution"
    assert harness_event.boundary_type == "harness_step"
    assert tool_event.side_effects is True


def test_partner_journal_comparison_endpoint(client: TestClient, poc_response: dict) -> None:
    run_id = poc_response["run_id"]
    trace = client.get(f"/debug/tasks/{run_id}/trace")
    assert trace.status_code == 200
    payload = trace.json()
    assert payload.get("run_id") == run_id
    assert len(payload.get("trace_events") or []) >= 1


def test_partner_sample_handoff_request_format(client: TestClient) -> None:
    assert _SAMPLE_REQUEST.is_file()
    body = json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8"))
    response = client.post(f"{_PREFIX}/poc/run", json=body)
    assert response.status_code == 200
    assert len(response.json().get("boundary_events") or []) >= 2
