# © Artur Czarnecki. All rights reserved.

"""Unit tests for EBE-9 host attestation sealing and verification."""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.canonical_json import canonical_json_text, stable_payload_hash
from intergrax.runtime.attestation.execution_boundary_event import (
    ExecutionBoundaryEventV1,
    ExecutionBoundaryLineageV1,
    ExecutionBoundaryRuntimeRefV1,
)
from intergrax.runtime.attestation.host_attestation import (
    HOST_ATTESTATION_CONTEXT,
    POC_ATTESTATION_DEMO_SIGNING_SEED,
    HostAttestationEnvelopeV1,
    build_host_attestation_sealer,
    statement_from_envelope,
    verify_host_attestation,
)

pytestmark = [pytest.mark.unit]

_GOLDEN_SIGNED_AT = "2026-06-19T12:00:00+00:00"


def _golden_tool_event() -> ExecutionBoundaryEventV1:
    return ExecutionBoundaryEventV1(
        event_id="4d9b7c34-ff54-4451-b6d3-54402c265715",
        event_sequence=1,
        boundary_type="tool_execution",
        tool_id="records.put",
        agent_id="boundary_demo_agent",
        run_id="run_golden_ebe9_vector",
        step_id="store_demo_record",
        task_id="run_golden_ebe9_vector",
        tenant_id="default",
        action_status="executed",
        side_effects=True,
        risk_level="medium",
        input={"partition_key": "attestation_demo", "row_key": "golden-001", "data": {"title": "Golden", "version": 1}},
        output={"stored": True, "partition_key": "attestation_demo", "row_key": "golden-001"},
        input_hash="sha256:1111111111111111111111111111111111111111111111111111111111111111",
        output_hash="sha256:2222222222222222222222222222222222222222222222222222222222222222",
        occurred_at=_GOLDEN_SIGNED_AT,
        lineage=ExecutionBoundaryLineageV1(ref="run_golden_ebe9_vector:store_demo_record"),
        runtime_ref=ExecutionBoundaryRuntimeRefV1(runtime_version="0.1.0"),
    )


@pytest.fixture
def demo_sealer():
    return build_host_attestation_sealer(
        public_key_id="attestation-demo-host-1",
        private_key_material=POC_ATTESTATION_DEMO_SIGNING_SEED,
    )


@pytest.fixture
def alt_sealer():
    alt_seed = Ed25519PrivateKey.generate().private_bytes_raw()
    return build_host_attestation_sealer(
        public_key_id="wrong-key-id",
        private_key_material=alt_seed,
    )


def test_host_attestation_valid_signature_round_trip(demo_sealer) -> None:
    event = _golden_tool_event()
    signed_event, envelope = demo_sealer.seal_event(event, signed_at=_GOLDEN_SIGNED_AT)
    assert signed_event.signed is True
    verify_host_attestation(signed_event, envelope, public_key=demo_sealer.public_key_bytes)
    statement = statement_from_envelope(envelope)
    assert statement["context"] == HOST_ATTESTATION_CONTEXT
    assert envelope.signed_payload_hash == stable_payload_hash(
        signed_event.model_copy(update={"signed": False}).model_dump(mode="json")
    )


def test_host_attestation_tampered_event_fails_verification(demo_sealer) -> None:
    event = _golden_tool_event()
    signed_event, envelope = demo_sealer.seal_event(event, signed_at=_GOLDEN_SIGNED_AT)
    tampered = signed_event.model_copy(update={"action_status": "failed"})
    with pytest.raises(ValueError, match="signed_payload_hash"):
        verify_host_attestation(tampered, envelope, public_key=demo_sealer.public_key_bytes)


def test_host_attestation_wrong_key_fails_verification(demo_sealer, alt_sealer) -> None:
    event = _golden_tool_event()
    _, envelope = demo_sealer.seal_event(event, signed_at=_GOLDEN_SIGNED_AT)
    with pytest.raises(Exception):
        verify_host_attestation(event, envelope, public_key=alt_sealer.public_key_bytes)


def test_host_attestation_tampered_envelope_hash_fails_verification(demo_sealer) -> None:
    event = _golden_tool_event()
    _, envelope = demo_sealer.seal_event(event, signed_at=_GOLDEN_SIGNED_AT)
    mutated = envelope.model_copy(
        update={"signed_payload_hash": "sha256:" + "ab" * 32},
    )
    with pytest.raises(ValueError, match="signed_payload_hash"):
        verify_host_attestation(event, mutated, public_key=demo_sealer.public_key_bytes)


def test_boundary_buffer_unsigned_compatibility() -> None:
    buffer = BoundaryEventBuffer(host_attestation_sealer=None)
    event = _golden_tool_event()
    buffer.append("run_unsigned", event)
    payload = buffer.snapshot_for_run("run_unsigned")[0]
    assert payload["signed"] is False
    assert payload["host_attestation"] is None


def test_boundary_buffer_host_signing_enabled(demo_sealer) -> None:
    buffer = BoundaryEventBuffer(host_attestation_sealer=demo_sealer)
    event = _golden_tool_event()
    buffer.append("run_signed", event)
    payload = buffer.snapshot_for_run("run_signed")[0]
    assert payload["signed"] is True
    assert payload["host_attestation"] is not None
    verify_host_attestation(
        {key: value for key, value in payload.items() if key != "host_attestation"},
        payload["host_attestation"],
        public_key=demo_sealer.public_key_bytes,
    )


def test_golden_vector_statement_canonical_bytes(demo_sealer) -> None:
    event = _golden_tool_event()
    _, envelope = demo_sealer.seal_event(event, signed_at=_GOLDEN_SIGNED_AT)
    statement = statement_from_envelope(envelope)
    canonical = canonical_json_text(statement)
    assert '"context":"boundaryattest.host-attestation.v1"' in canonical
    assert '"public_key_id":"attestation-demo-host-1"' in canonical
    signature = base64.b64decode(envelope.signature.encode("ascii"))
    demo_sealer._private_key.public_key().verify(
        signature,
        canonical.encode("utf-8"),
    )
