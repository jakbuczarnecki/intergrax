# © Artur Czarnecki. All rights reserved.

"""Acceptance tests aligned with AgentReceipt partner PoC agreement.

Validates:
- execution-boundary event delivered in trigger API response (no webhook)
- required event fields for AgentReceipt mapping
- unsigned events + honest trust_model (client_observed, not server attested)
- records.put tool boundary + journal comparison endpoint
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from attestation_demo.host.factory import create_attestation_demo_application
from attestation_demo.partner_handoff.contract_assertions import assert_partner_poc_response_shape

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


def test_partner_delivery_via_trigger_api_response_not_webhook(poc_response: dict) -> None:
    """PoC v1: boundary event in POST /poc/run response; webhook deferred."""
    assert_partner_poc_response_shape(poc_response)
    assert poc_response.get("state") == "completed"


def test_partner_agent_and_tool_identity(poc_response: dict) -> None:
    event = poc_response["boundary_events"][0]
    assert poc_response.get("agent_id") == "boundary_demo_agent"
    assert event["agent_id"] == "boundary_demo_agent"
    assert event["tool_id"] == "records.put"


def test_partner_executed_status_and_canonical_io(poc_response: dict) -> None:
    event = poc_response["boundary_events"][0]
    assert event["action_status"] == "executed"
    assert isinstance(event["input"], dict)
    assert isinstance(event["output"], dict)
    assert event["input"].get("partition_key") == "attestation_demo"
    assert event["input"].get("data", {}).get("title") == "PoC report"
    assert event["output"].get("stored") is True


def test_partner_optional_hashes_for_cross_check(poc_response: dict) -> None:
    event = poc_response["boundary_events"][0]
    assert event.get("input_hash", "").startswith("sha256:")
    assert event.get("output_hash", "").startswith("sha256:")


def test_partner_run_id_step_id_lineage_and_timestamp(poc_response: dict) -> None:
    run_id = poc_response["run_id"]
    event = poc_response["boundary_events"][0]
    assert run_id
    assert event["run_id"] == run_id
    assert event["step_id"] == "store_demo_record"
    assert event["lineage"]["type"] == "execution_record"
    assert run_id in event["lineage"]["ref"]
    assert event.get("occurred_at")


def test_partner_unsigned_event_no_platform_attestation(poc_response: dict) -> None:
    event = poc_response["boundary_events"][0]
    assert event["signed"] is False
    trust = poc_response.get("trust_model") or {}
    assert trust.get("platform_signed") == "false"
    assert trust.get("recommended_receipt_role") == "client_observed"
    note = (trust.get("note") or "").lower()
    assert "unsigned" in note or "partner signs" in note
    assert "server_attested" not in json.dumps(trust).lower()


def test_partner_records_put_tool_boundary_not_step_level_kernel(poc_response: dict) -> None:
    event = poc_response["boundary_events"][0]
    assert event["boundary_type"] == "tool_execution"
    assert event["tool_id"] == "records.put"
    assert event["side_effects"] is True


def test_partner_journal_comparison_endpoint(client: TestClient, poc_response: dict) -> None:
    """Adapter flow: compare receipt with Intergrax journal for same run_id."""
    run_id = poc_response["run_id"]
    trace = client.get(f"/debug/tasks/{run_id}/trace")
    assert trace.status_code == 200
    payload = trace.json()
    assert payload.get("run_id") == run_id
    assert len(payload.get("trace_events") or []) >= 1


def test_partner_sample_handoff_request_format(client: TestClient) -> None:
    """Committed partner_handoff/poc_run_request.v1.json must trigger successfully."""
    assert _SAMPLE_REQUEST.is_file()
    body = json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8"))
    response = client.post(f"{_PREFIX}/poc/run", json=body)
    assert response.status_code == 200
    assert response.json().get("boundary_events")
