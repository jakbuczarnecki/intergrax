# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.payload_registry import merge_payload_envelope, validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import HumanPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.spine_payload_codec import (
    legacy_spine_payload_to_typed,
    prepare_canonical_production_write_event,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase

pytestmark = pytest.mark.gate


def test_human_payload_roundtrip_nested_human_request() -> None:
    raw = {
        "human_request": {
            "request_id": "req-A",
            "prompt": "approve?",
            "options": ["approve", "reject"],
            "blocking": True,
        }
    }
    typed, _ = legacy_spine_payload_to_typed(RuntimeEventType.HUMAN_APPROVAL_REQUESTED, raw)
    assert isinstance(typed, HumanPayloadV1)
    assert typed.request_id == "req-A"
    envelope = merge_payload_envelope(raw, typed)
    decoded = validate_payload_envelope(envelope)
    assert decoded is not None
    assert decoded.request_id == "req-A"


def test_human_payload_preserves_two_distinct_request_ids() -> None:
    for request_id in ("req-A", "req-B"):
        typed, _ = legacy_spine_payload_to_typed(
            RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
            {"human_request": {"request_id": request_id, "options": ["approve"]}},
        )
        assert typed.request_id == request_id
        envelope = merge_payload_envelope({}, typed)
        decoded = validate_payload_envelope(envelope)
        assert decoded is not None
        assert decoded.request_id == request_id


def test_human_payload_missing_request_id_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing request_id"):
        legacy_spine_payload_to_typed(
            RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
            {"human_request": {"prompt": "no id"}},
        )


def test_human_approval_canonical_write_pipeline_preserves_request_id() -> None:
    event = RuntimeEvent(
        tenant_id="tenant",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
        phase=ExecutionPhase.HUMAN_APPROVAL,
        message="human approval requested",
        payload={
            "human_request": {
                "request_id": "req-pipeline",
                "prompt": "approve?",
                "options": ["approve"],
                "blocking": True,
            }
        },
    )
    prepared = prepare_canonical_production_write_event(event)
    decoded = validate_payload_envelope(prepared.payload)
    assert decoded is not None
    assert decoded.request_id == "req-pipeline"


def test_human_payload_decode_does_not_mint_request_id() -> None:
    typed = HumanPayloadV1(request_id="req-fixed", option_selected="pending")
    envelope = merge_payload_envelope({}, typed)
    decoded = validate_payload_envelope(envelope)
    assert decoded is not None
    assert decoded.request_id == "req-fixed"
