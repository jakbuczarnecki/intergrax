# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.payload_registry import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    RuntimeEventPayloadError,
    UnknownPayloadSchemaError,
    assert_canonical_production_runtime_event_payload,
    get_payload_schema,
    list_registered_payload_schema_ids,
    merge_payload_envelope,
    register_payload_schema,
    runtime_event_with_payload,
    validate_payload_envelope,
)
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.events.payloads import (
    CANONICAL_PAYLOAD_TYPES,
    SkillResolvedPayloadV1,
    ToolPayloadV1,
)
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
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


def test_validate_envelope_accepts_matching_version_from_to_envelope() -> None:
    typed = ToolPayloadV1(tool_name="websearch.query", status="requested")
    validate_payload_envelope(typed.to_envelope())


def test_validate_envelope_rejects_missing_payload_schema_version() -> None:
    typed = ToolPayloadV1(tool_name="websearch.query", status="requested")
    envelope = typed.to_envelope()
    del envelope["payload_schema_version"]
    with pytest.raises(RuntimeEventPayloadError, match="payload_schema_version"):
        validate_payload_envelope(envelope)


def test_validate_envelope_rejects_wrong_numeric_version() -> None:
    schema_cls = ToolPayloadV1
    with pytest.raises(RuntimeEventPayloadError, match="mismatch"):
        validate_payload_envelope(
            {
                "payload_schema_id": schema_cls.schema_id,
                "payload_schema_version": schema_cls.schema_version + 1,
                "data": {"tool_name": "t", "status": "requested"},
            }
        )


@pytest.mark.parametrize(
    "bad_version",
    ["1", True, None],
)
def test_validate_envelope_rejects_wrong_type_version(bad_version: object) -> None:
    with pytest.raises(RuntimeEventPayloadError, match="payload_schema_version"):
        validate_payload_envelope(
            {
                "payload_schema_id": ToolPayloadV1.schema_id,
                "payload_schema_version": bad_version,
                "data": {"tool_name": "t", "status": "requested"},
            }
        )


def test_validate_envelope_rejects_correct_id_wrong_version() -> None:
    with pytest.raises(RuntimeEventPayloadError, match="mismatch"):
        validate_payload_envelope(
            {
                "payload_schema_id": "tool.v1",
                "payload_schema_version": 99,
                "data": {"tool_name": "t", "status": "requested"},
            }
        )


def test_validate_envelope_unknown_schema_id_v6() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {
                "payload_schema_id": "unknown.compat_x.v1",
                "payload_schema_version": 1,
                "data": {},
            }
        )


def test_validate_envelope_allows_legacy_dict_without_schema() -> None:
    assert validate_payload_envelope({"tool_name": "legacy"}) is None


def test_runtime_event_with_payload_merges_envelope() -> None:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    event = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
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


def test_canonical_production_write_rejects_legacy_untyped_payload() -> None:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    event = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    with pytest.raises(RuntimeEventPayloadError):
        assert_canonical_production_runtime_event_payload(event)
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(event)


def test_validating_persistence_upgrades_legacy_plan_payload() -> None:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
    from intergrax.runtime.events.stores.validating_runtime_event_store import (
        ValidatingRuntimeEventPersistence,
    )

    inner = InMemoryRuntimeEventStore()
    store = ValidatingRuntimeEventPersistence(inner)
    event = RuntimeEvent(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.PLAN_CREATED,
        phase=ExecutionPhase.PLANNING,
        payload={"plan_id": "plan-1", "step_count": 2, "task_state": "planned"},
    )
    positioned = store.append(event, tenant_id="tenant")
    assert positioned.event.payload["payload_schema_id"] == "plan_lifecycle.v1"
    validate_payload_envelope(positioned.event.payload)


def test_merge_payload_envelope_promotes_ops_fields() -> None:
    typed = ToolPayloadV1(tool_name="websearch.query", status="requested")
    merged = merge_payload_envelope({}, typed, promote_fields={"tool_name": "websearch.query"})
    assert merged["tool_name"] == "websearch.query"
    assert merged["payload_schema_id"] == "tool.v1"
