# © Artur Czarnecki. All rights reserved.

"""EBE-9 host signing acceptance tests for BoundaryAttest partner handoff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from attestation_demo.host.factory import create_attestation_demo_application
from attestation_demo.manifest import build_attestation_demo_environment
from attestation_demo.partner_handoff.contract_assertions import (
    assert_partner_failed_tool_dual_claims,
    assert_partner_host_attestation_envelope,
    assert_partner_poc_response_shape,
    partner_host_signing_enabled,
)
from intergrax.applications._shared.attestation_runtime_bridge import build_boundary_event_buffer
from intergrax.applications.contracts.environment_profile import ExecutionBoundaryExportProfile
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.canonical_json import canonical_json_text
from intergrax.runtime.attestation.host_attestation import (
    POC_ATTESTATION_DEMO_SIGNING_SEED,
    build_host_attestation_sealer,
)

_PREFIX = "/v1/attestation_demo"
_HANDOFF_DIR = Path(__file__).resolve().parents[2] / "partner_handoff"
_SAMPLE_REQUEST = _HANDOFF_DIR / "poc_run_request.v1.json"
_GOLDEN_VECTOR = _HANDOFF_DIR / "ebe9_golden_vector.v1.json"

pytestmark = [pytest.mark.unit]


@pytest.fixture
def pinned_public_key() -> bytes:
    sealer = build_host_attestation_sealer(
        public_key_id="attestation-demo-host-1",
        private_key_material=POC_ATTESTATION_DEMO_SIGNING_SEED,
    )
    assert sealer is not None
    return sealer.public_key_bytes


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_attestation_demo_application())


@pytest.fixture
def poc_response(client: TestClient) -> dict:
    body = json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8"))
    response = client.post(f"{_PREFIX}/poc/run", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_ebe9_host_signing_enabled_on_default_manifest(poc_response: dict) -> None:
    assert partner_host_signing_enabled(poc_response["boundary_events"])
    assert poc_response["trust_model"]["platform_signed"] == "true"
    assert poc_response["trust_model"]["recommended_receipt_role"] == "host_attested"


def test_ebe9_one_signature_per_event_not_composite(poc_response: dict, pinned_public_key: bytes) -> None:
    assert_partner_poc_response_shape(poc_response)
    events = poc_response["boundary_events"]
    assert len(events) == 2
    hashes = {event["host_attestation"]["signed_payload_hash"] for event in events}
    assert len(hashes) == 2
    for event in events:
        assert event["signed"] is True
        assert_partner_host_attestation_envelope(event, public_key=pinned_public_key)


def test_ebe9_failed_tool_and_completed_harness_both_signed(pinned_public_key: bytes) -> None:
    class _FailingPutDocumentStore(InMemoryDocumentStore):
        def put(self, document: DocumentRecord) -> None:
            raise RuntimeError("simulated_storage_failure")

    client = TestClient(
        create_attestation_demo_application(document_store=_FailingPutDocumentStore()),
    )
    response = client.post(
        f"{_PREFIX}/poc/run",
        json={
            "message": "ebe9 failure path",
            "capability": "attestation.demo",
            "partition_key": "attestation_demo",
            "row_key": "poc-ebe9-fail-001",
            "record_data": {"title": "fail", "version": 1},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert_partner_failed_tool_dual_claims(
        payload["boundary_events"],
        run_id=str(payload.get("run_id") or ""),
        host_signed=True,
    )
    for event in payload["boundary_events"]:
        assert_partner_host_attestation_envelope(event, public_key=pinned_public_key)


def test_ebe9_unsigned_v2_still_supported_when_host_signing_disabled() -> None:
    env = build_attestation_demo_environment().model_copy(
        update={
            "execution_boundary_export_profile": ExecutionBoundaryExportProfile(
                enabled=True,
                capture_mode="side_effects_only",
                include_canonical_io=True,
                step_level_enabled=True,
                host_signing_enabled=False,
            ),
        }
    )
    unsigned_buffer = build_boundary_event_buffer(env) or BoundaryEventBuffer()
    assert unsigned_buffer.host_signing_enabled is False
    client = TestClient(
        create_attestation_demo_application(boundary_event_buffer=unsigned_buffer),
    )
    response = client.post(
        f"{_PREFIX}/poc/run",
        json=json.loads(_SAMPLE_REQUEST.read_text(encoding="utf-8")),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trust_model"]["platform_signed"] == "false"
    assert payload["trust_model"]["recommended_receipt_role"] == "client_observed"
    for event in payload["boundary_events"]:
        assert event["signed"] is False
        assert event["host_attestation"] is None


def test_ebe9_golden_vector_file_matches_runtime_crypto(pinned_public_key: bytes) -> None:
    assert _GOLDEN_VECTOR.is_file(), "partner_handoff/ebe9_golden_vector.v1.json required"
    vector = json.loads(_GOLDEN_VECTOR.read_text(encoding="utf-8"))
    assert vector["public_key_id"] == "attestation-demo-host-1"
    assert vector["public_key_ed25519"] == base64_public_key(pinned_public_key)
    assert_partner_host_attestation_envelope(
        {
            **vector["event"],
            "signed": True,
            "host_attestation": vector["host_attestation"],
        },
        public_key=pinned_public_key,
    )
    assert vector["signed_payload_hash"] == vector["host_attestation"]["signed_payload_hash"]
    assert vector["canonical_statement"] == canonical_json_text(
        vector["host_attestation_statement"],
    )


def base64_public_key(key_bytes: bytes) -> str:
    import base64

    return base64.b64encode(key_bytes).decode("ascii")
