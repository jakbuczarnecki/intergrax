# © Artur Czarnecki. All rights reserved.

"""Partner PoC contract assertions (shared by tests and handoff verification)."""

from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1
from intergrax.runtime.attestation.host_attestation import (
    HOST_ATTESTATION_CONTEXT,
    HOST_ATTESTATION_ENVELOPE_SCHEMA,
    HostAttestationEnvelopeV1,
    verify_host_attestation,
)

PARTNER_REQUIRED_EVENT_FIELDS = frozenset(
    {
        "schema_id",
        "signed",
        "event_id",
        "event_sequence",
        "boundary_type",
        "agent_id",
        "run_id",
        "step_id",
        "action_status",
        "input",
        "output",
        "input_hash",
        "output_hash",
        "occurred_at",
        "lineage",
        "host_attestation",
    }
)

PARTNER_TOOL_EVENT_FIELDS = PARTNER_REQUIRED_EVENT_FIELDS | frozenset(
    {"tool_id", "side_effects"}
)

PARTNER_HARNESS_EVENT_FIELDS = PARTNER_REQUIRED_EVENT_FIELDS | frozenset(
    {"policy_verdicts", "step_outcome"}
)

PARTNER_REQUIRED_RESPONSE_FIELDS = frozenset(
    {
        "task_id",
        "run_id",
        "state",
        "agent_id",
        "boundary_events",
        "trust_model",
    }
)


def partner_host_signing_enabled(events: list[dict[str, Any]]) -> bool:
    return any(event.get("signed") is True for event in events)


def assert_partner_host_attestation_envelope(
    event: dict[str, Any],
    *,
    public_key: Ed25519PublicKey | bytes | None = None,
    expect_public_key_id: str = "attestation-demo-host-1",
) -> HostAttestationEnvelopeV1:
    assert event.get("signed") is True, "host signing expected"
    envelope_raw = event.get("host_attestation")
    assert envelope_raw is not None, "host_attestation envelope required when signed=true"
    envelope = HostAttestationEnvelopeV1.model_validate(envelope_raw)
    assert envelope.schema_id == HOST_ATTESTATION_ENVELOPE_SCHEMA
    assert envelope.context == HOST_ATTESTATION_CONTEXT
    assert envelope.payload_schema_id == "execution_boundary_event.v1"
    assert envelope.signature_algorithm == "Ed25519"
    assert envelope.public_key_id == expect_public_key_id
    assert envelope.signed_payload_hash.startswith("sha256:")
    assert envelope.signature
    base64.b64decode(envelope.signature.encode("ascii"))
    if public_key is not None:
        verify_host_attestation(event, envelope, public_key=public_key)
    return envelope


def assert_partner_poc_response_shape(payload: dict[str, Any]) -> None:
    missing = PARTNER_REQUIRED_RESPONSE_FIELDS - set(payload.keys())
    assert not missing, f"missing response fields: {sorted(missing)}"
    trust = payload.get("trust_model") or {}
    events = payload.get("boundary_events") or []
    assert events, "boundary_events must be non-empty"
    host_signed = partner_host_signing_enabled(events)
    if host_signed:
        assert trust.get("platform_signed") == "true"
        assert trust.get("recommended_receipt_role") == "host_attested"
    else:
        assert trust.get("platform_signed") == "false"
        assert trust.get("recommended_receipt_role") == "client_observed"
    assert "server_attested" not in json.dumps(trust).lower()
    assert_partner_poc_v2_events(events, run_id=str(payload.get("run_id") or ""), host_signed=host_signed)


def assert_partner_poc_v2_events(
    events: list[dict[str, Any]],
    *,
    run_id: str = "",
    host_signed: bool = False,
) -> None:
    assert len(events) >= 2, "PoC v2 expects tool_execution + harness_step events"
    sequences = [event.get("event_sequence") for event in events]
    assert sequences == sorted(sequences), "events must be ordered by event_sequence"
    assert len(set(sequences)) == len(sequences), "event_sequence must be unique per run"
    event_ids = [str(event.get("event_id") or "") for event in events]
    assert all(event_ids), "each event must have event_id"
    assert len(set(event_ids)) == len(event_ids), "event_id must be unique per run"
    boundary_types = {event.get("boundary_type") for event in events}
    assert boundary_types == {"tool_execution", "harness_step"}
    tool_events = [event for event in events if event.get("boundary_type") == "tool_execution"]
    harness_events = [event for event in events if event.get("boundary_type") == "harness_step"]
    assert len(tool_events) == 1
    assert len(harness_events) == 1
    assert_partner_tool_boundary_event(
        tool_events[0],
        run_id=run_id,
        host_signed=host_signed,
    )
    assert_partner_harness_boundary_event(
        harness_events[0],
        run_id=run_id,
        host_signed=host_signed,
    )


def assert_partner_failed_tool_dual_claims(
    events: list[dict[str, Any]],
    *,
    run_id: str = "",
    host_signed: bool = False,
) -> None:
    """Tool failure still emits two distinct boundary claims (enterprise separation)."""
    assert_partner_poc_v2_events(events, run_id=run_id, host_signed=host_signed)
    tool_event = next(event for event in events if event.get("boundary_type") == "tool_execution")
    harness_event = next(event for event in events if event.get("boundary_type") == "harness_step")
    assert tool_event.get("action_status") == "failed"
    assert tool_event.get("error_message")
    assert harness_event.get("action_status") == "completed"
    assert (harness_event.get("step_outcome") or {}).get("outcome_applied") is True


def assert_partner_tool_boundary_event(
    event: dict[str, Any],
    *,
    run_id: str = "",
    host_signed: bool = False,
    verify_public_key: Ed25519PublicKey | bytes | None = None,
) -> ExecutionBoundaryEventV1:
    missing = PARTNER_TOOL_EVENT_FIELDS - set(event.keys())
    assert not missing, f"missing tool event fields: {sorted(missing)}"
    assert event.get("schema_id") == "execution_boundary_event.v1"
    assert event.get("boundary_type") == "tool_execution"
    assert event.get("tool_id") == "records.put"
    assert event.get("agent_id") == "boundary_demo_agent"
    assert event.get("action_status") in {"executed", "failed"}
    assert str(event.get("input_hash", "")).startswith("sha256:")
    if event.get("action_status") == "failed":
        assert event.get("error_message")
    else:
        assert str(event.get("output_hash", "")).startswith("sha256:")
    if run_id:
        assert event.get("run_id") == run_id
        assert run_id in str((event.get("lineage") or {}).get("ref", ""))
    if host_signed:
        assert event.get("signed") is True
        assert_partner_host_attestation_envelope(event, public_key=verify_public_key)
    else:
        assert event.get("signed") is False
        assert event.get("host_attestation") is None
    return ExecutionBoundaryEventV1.model_validate(
        {key: value for key, value in event.items() if key != "host_attestation"}
    )


def assert_partner_harness_boundary_event(
    event: dict[str, Any],
    *,
    run_id: str = "",
    host_signed: bool = False,
    verify_public_key: Ed25519PublicKey | bytes | None = None,
) -> ExecutionBoundaryEventV1:
    missing = PARTNER_HARNESS_EVENT_FIELDS - set(event.keys())
    assert not missing, f"missing harness event fields: {sorted(missing)}"
    assert event.get("schema_id") == "execution_boundary_event.v1"
    assert event.get("boundary_type") == "harness_step"
    assert event.get("agent_id") == "boundary_demo_agent"
    assert event.get("action_status") in {"completed", "denied", "failed", "paused"}
    assert event.get("tool_id") is None
    assert isinstance(event.get("policy_verdicts"), list)
    assert isinstance(event.get("step_outcome"), dict)
    assert str(event.get("input_hash", "")).startswith("sha256:")
    assert str(event.get("output_hash", "")).startswith("sha256:")
    if run_id:
        assert event.get("run_id") == run_id
        assert run_id in str((event.get("lineage") or {}).get("ref", ""))
    if host_signed:
        assert event.get("signed") is True
        assert_partner_host_attestation_envelope(event, public_key=verify_public_key)
    else:
        assert event.get("signed") is False
        assert event.get("host_attestation") is None
    return ExecutionBoundaryEventV1.model_validate(
        {key: value for key, value in event.items() if key != "host_attestation"}
    )


def assert_partner_boundary_event(
    event: dict[str, Any],
    *,
    run_id: str = "",
    host_signed: bool = False,
    verify_public_key: Ed25519PublicKey | bytes | None = None,
) -> ExecutionBoundaryEventV1:
    boundary_type = event.get("boundary_type")
    if boundary_type == "harness_step":
        return assert_partner_harness_boundary_event(
            event,
            run_id=run_id,
            host_signed=host_signed,
            verify_public_key=verify_public_key,
        )
    return assert_partner_tool_boundary_event(
        event,
        run_id=run_id,
        host_signed=host_signed,
        verify_public_key=verify_public_key,
    )
