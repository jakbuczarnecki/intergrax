# © Artur Czarnecki. All rights reserved.

"""Regression tests for partner PoC contract hardening."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from attestation_demo.partner_handoff.contract_assertions import (
    assert_partner_boundary_event,
    assert_partner_failed_tool_dual_claims,
    assert_partner_poc_response_shape,
)
from attestation_demo.host.factory import create_attestation_demo_application
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/attestation_demo"
_HANDOFF_DIR = Path(__file__).resolve().parents[2] / "partner_handoff"
_SAMPLE_REQUEST = _HANDOFF_DIR / "poc_run_request.v1.json"


class _FailingPutDocumentStore(InMemoryDocumentStore):
    """Lab store that fails on put — exercises failed boundary-event path."""

    def put(self, document: DocumentRecord) -> None:
        raise RuntimeError("simulated_storage_failure")


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_attestation_demo_application())


def test_partner_openapi_has_no_webhook_delivery_routes(client: TestClient) -> None:
    """PoC v1 defers webhook; surface must not advertise webhook sinks."""
    spec = client.get("/openapi.json").json()
    paths = "\n".join(spec.get("paths", {}).keys()).lower()
    assert "webhook" not in paths
    assert "/poc/run" in paths


def test_partner_boundary_event_validates_against_pydantic_schema(client: TestClient) -> None:
    body = json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8"))
    response = client.post(f"{_PREFIX}/poc/run", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert_partner_poc_response_shape(payload)


def test_partner_failed_tool_returns_dual_boundary_claims() -> None:
    """Tool failure: separate tool receipt (failed) and harness receipt (completed)."""
    client = TestClient(
        create_attestation_demo_application(document_store=_FailingPutDocumentStore()),
    )
    response = client.post(
        f"{_PREFIX}/poc/run",
        json={
            "message": "forced failure path",
            "capability": "attestation.demo",
            "partition_key": "attestation_demo",
            "row_key": "poc-fail-001",
            "record_data": {"title": "fail case", "version": 1},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    events = payload.get("boundary_events") or []
    assert_partner_failed_tool_dual_claims(events, run_id=str(payload.get("run_id") or ""))
    trust = payload.get("trust_model") or {}
    assert trust.get("recommended_receipt_role") == "client_observed"
