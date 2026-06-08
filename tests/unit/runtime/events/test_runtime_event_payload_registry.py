# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.payload_registry import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    UnknownPayloadSchemaError,
    get_payload_schema,
    list_registered_payload_schema_ids,
    merge_payload_envelope,
    register_payload_schema,
    runtime_event_with_payload,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads import (
    CANONICAL_PAYLOAD_TYPES,
    SkillResolvedPayloadV1,
    ToolPayloadV1,
)
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase

pytestmark = pytest.mark.gate


def test_canonical_payload_families_registered() -> None:
    registered = set(list_registered_payload_schema_ids(include_extensions=False))
    for schema_cls in CANONICAL_PAYLOAD_TYPES:
        assert schema_cls.schema_id in registered
        assert get_payload_schema(schema_cls.schema_id) is schema_cls


def test_tool_payload_round_trip_envelope() -> None:
    typed = ToolPayloadV1(
        tool_name="rag.retrieve",
        status="completed",
        duration_ms=42,
        redacted_input_summary="[REDACTED]",
        step_id="rag_step",
    )
    envelope = typed.to_envelope()
    assert envelope["payload_schema_id"] == "tool.v1"
    assert envelope["payload_schema_version"] == 1
    parsed = validate_payload_envelope(envelope)
    assert isinstance(parsed, ToolPayloadV1)
    assert parsed.tool_name == "rag.retrieve"
    assert parsed.duration_ms == 42
    restored = ToolPayloadV1.from_envelope(envelope)
    assert restored == typed


def test_validate_envelope_rejects_unknown_schema() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {
                "payload_schema_id": "unknown.schema.v99",
                "payload_schema_version": 1,
                "data": {},
            }
        )


def test_validate_envelope_allows_legacy_dict_without_schema() -> None:
    assert validate_payload_envelope({"tool_name": "legacy"}) is None


def test_runtime_event_with_payload_merges_envelope() -> None:
    event = RuntimeEvent(
        task_id="task-1",
        run_id="run-1",
        event_type=RuntimeEventType.SKILL_RESOLVED,
        phase=ExecutionPhase.AGENT_SELECTION,
    )
    typed = SkillResolvedPayloadV1(
        skill_ids=("skill.a",),
        tool_ids=("tool.a",),
        prompt_instruction_ids=(),
        policy_fragment_ids=(),
        risk_tier="low",
    )
    merged = runtime_event_with_payload(event, typed)
    assert merged.payload["payload_schema_id"] == "skill_resolved.v1"
    assert merged.payload["data"]["skill_ids"] == ["skill.a"]
    validate_payload_envelope(merged.payload)


def test_extension_payload_registration() -> None:
    class _AgentCustomPayload(RuntimeEventPayload):
        schema_id = "agents.test.diag.custom"

        detail: str

    register_payload_schema(_AgentCustomPayload, extension=True)
    assert get_payload_schema("agents.test.diag.custom") is _AgentCustomPayload


def test_preferred_schema_ids_reference_registered_types() -> None:
    for event_type, schema_id in EVENT_TYPE_PREFERRED_SCHEMA.items():
        assert get_payload_schema(schema_id) is not None, f"{event_type.value} -> {schema_id}"


def test_merge_payload_envelope_promotes_ops_fields() -> None:
    typed = ToolPayloadV1(tool_name="websearch.query", status="requested")
    merged = merge_payload_envelope({}, typed, promote_fields={"tool_name": "websearch.query"})
    assert merged["tool_name"] == "websearch.query"
    assert merged["payload_schema_id"] == "tool.v1"
