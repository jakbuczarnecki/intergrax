"""NT-RW-GATES-RTEV — RuntimeEvent schema version and typed payload consumer alignment."""

from __future__ import annotations

import pytest
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    RuntimeEventType,
    parse_runtime_event_payload,
)
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    bootstrap_canonical_payload_registry,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads.canonical import ToolPayloadV1

pytestmark = pytest.mark.unit


def _minimal_event_dict() -> dict[str, object]:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_event_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    return {
        "event_id": mint_event_id(),
        "tenant_id": "t1",
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "event_type": RuntimeEventType.TOOL_COMPLETED.value,
        "phase": ExecutionPhase.STEP_EXECUTION.value,
        "schema_version": RUNTIME_EVENT_SCHEMA_VERSION,
        "correlation_id": "c1",
    }


def test_parse_runtime_event_rejects_missing_schema_version() -> None:
    data = _minimal_event_dict()
    del data["schema_version"]
    with pytest.raises(ValueError, match="missing runtime event schema_version"):
        parse_runtime_event_payload(data)


def test_parse_runtime_event_rejects_retired_v1() -> None:
    data = _minimal_event_dict()
    data["schema_version"] = "runtime_event.v1"
    with pytest.raises(ValueError, match="retired"):
        parse_runtime_event_payload(data)


def test_parse_runtime_event_rejects_unknown_schema_version() -> None:
    data = _minimal_event_dict()
    data["schema_version"] = "runtime_event.v99"
    with pytest.raises(ValueError, match="unsupported runtime event schema version"):
        parse_runtime_event_payload(data)


def test_typed_payload_producer_consumer_round_trip() -> None:
    bootstrap_canonical_payload_registry()
    payload = ToolPayloadV1(tool_name="demo", status="ok", duration_ms=12)
    envelope = payload.to_envelope()
    restored = validate_payload_envelope(envelope)
    assert restored is not None
    assert isinstance(restored, ToolPayloadV1)
    assert restored.tool_name == "demo"


def test_typed_payload_schema_mismatch_denied() -> None:
    bootstrap_canonical_payload_registry()
    envelope = ToolPayloadV1(tool_name="demo", status="ok").to_envelope()
    envelope["payload_schema_id"] = "tool.v99"
    with pytest.raises(RuntimeEventPayloadError):
        validate_payload_envelope(envelope)


def test_runtime_event_round_trip_preserves_schema_version() -> None:
    event = RuntimeEvent.model_validate(_minimal_event_dict())
    restored = RuntimeEvent.model_validate(event.model_dump(mode="json"))
    assert restored.schema_version == RUNTIME_EVENT_SCHEMA_VERSION


def test_qualification_trace_schema_ids_align_with_trace_bridge_consumer() -> None:
    from intergrax.runtime.events.payloads.canonical import TraceBridgePayloadV1
    from intergrax.runtime.observability.qualification_runtime_trace import (
        TRACE_BRIDGE_PAYLOAD_SCHEMA_ID,
        qualification_runtime_trace_diagnostic_schema_ids,
    )

    assert TraceBridgePayloadV1.schema_id == TRACE_BRIDGE_PAYLOAD_SCHEMA_ID
    ids = qualification_runtime_trace_diagnostic_schema_ids()
    assert ids
    assert all(isinstance(item, str) and item for item in ids)
