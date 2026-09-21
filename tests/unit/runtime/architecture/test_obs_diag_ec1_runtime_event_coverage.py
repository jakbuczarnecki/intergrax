# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1 — strict-write codec coverage and negative write matrix."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    validate_payload_envelope,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.runtime_event_payload_policy import (
    CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES,
    PayloadWriteMode,
    RuntimeEventTypeClassification,
    get_runtime_event_payload_policy,
    iter_runtime_event_payload_policies,
)
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.events.event_kind import DomainSignalError, validate_event_kind
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.evidence_persistence_adapter import (
    RuntimeEventPersistenceEvidenceAdapter,
)
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.payloads.canonical import (
    AgentSelectionPayloadV1,
    ToolPayloadV1,
)
from intergrax.runtime.events.spine_payload_codec import (
    legacy_spine_payload_to_typed,
    prepare_canonical_production_write_event,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.events.validating_evidence_persistence_port import (
    ValidatingEvidencePersistencePort,
)
from intergrax.runtime.observability.extension_sdk import (
    ExtensionSchemaError,
    register_extension_runtime_payload,
)
from pydantic import Field
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXEC_ID = mint_execution_id()


class _AppendCountingStore(InMemoryRuntimeEventStore):
    def __init__(self) -> None:
        super().__init__()
        self.append_calls = 0

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> object:
        self.append_calls += 1
        return super().append(event, tenant_id=tenant_id)


def _tool_requested_event(**payload: object) -> RuntimeEvent:
    identity = runtime_event_test_identity()
    return RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload=dict(payload),
    )


def _assert_invalid_canonical_write_rejects_before_provider(event: RuntimeEvent) -> None:
    inner = _AppendCountingStore()
    store = ValidatingRuntimeEventPersistence(inner)
    with pytest.raises(RuntimeEventSchemaError):
        store.append(event, tenant_id="tenant-ec1")
    assert inner.append_calls == 0

    evidence_inner = _AppendCountingStore()
    port = ValidatingEvidencePersistencePort(
        RuntimeEventPersistenceEvidenceAdapter(evidence_inner)
    )
    with pytest.raises(RuntimeEventSchemaError):
        port.append(event, tenant_id="tenant-ec1")
    assert evidence_inner.append_calls == 0

_REPRESENTATIVE_LEGACY_RAW: dict[RuntimeEventType, dict[str, Any]] = {
    RuntimeEventType.DECISION_EMITTED: {
        "decision": "continue",
        "reason": "policy allowed",
        "step_id": "step-1",
        "policy_action": "allow",
    },
    RuntimeEventType.AGENT_SELECTED: {"selected_agent_id": "agent.a"},
    RuntimeEventType.CONTEXT_BUILT: {
        "node_id": "n1",
        "context_original_chars": 10,
        "context_final_chars": 8,
    },
    RuntimeEventType.CONTEXT_ASSEMBLED: {
        "node_id": "n1",
        "context_original_chars": 10,
        "context_final_chars": 8,
    },
    RuntimeEventType.CONTEXT_TRIMMED: {
        "node_id": "n1",
        "context_original_chars": 10,
        "context_final_chars": 5,
        "trimmed": True,
    },
    RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED: {
        "provider_id": "provider.a",
        "fragment_count": 2,
    },
    RuntimeEventType.CONTEXT_CANDIDATE_DROPPED: {
        "provider_id": "provider.a",
        "drop_reason": "budget",
    },
    RuntimeEventType.EXECUTION_FAILED: {
        "failure_kind": "delegate_exception",
        "safe_summary": "failed",
    },
    RuntimeEventType.EXTERNAL_OPERATION_FAILED: {
        "execution_id": _EXEC_ID,
        "operation_attempt_id": "op-1",
        "provider_id": "provider",
        "operation_type": "llm",
        "failure_kind": "external_operation.timeout",
    },
    RuntimeEventType.HANDOFF_INITIATED: {
        "from_agent_id": "a1",
        "to_agent_id": "a2",
        "to_capability": "cap.b",
    },
    RuntimeEventType.HANDOFF_COMPLETED: {
        "from_agent_id": "a1",
        "to_agent_id": "a2",
    },
    RuntimeEventType.DELEGATION_GRANTED: {
        "parent_agent_id": "parent",
        "child_agent_id": "child",
        "node_id": "node-1",
    },
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: {
        "human_request": {"request_id": "req-1", "options": ["approve"]},
    },
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: {
        "human_request_id": "req-1",
        "verdict": "approve",
    },
    RuntimeEventType.VALIDATION_FAILED: {"errors": ["invalid output"]},
    RuntimeEventType.PLAN_CREATED: {
        "plan_id": "plan-1",
        "step_count": 1,
        "task_state": "planned",
    },
    RuntimeEventType.TOOL_REQUESTED: {"tool_name": "tool.test", "status": "requested"},
}


def test_runtime_event_payload_policy_covers_all_58_enum_members() -> None:
    assert {event_type for event_type, _ in iter_runtime_event_payload_policies()} == set(
        RuntimeEventType
    )


@pytest.mark.parametrize("event_type", sorted(CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES, key=lambda e: e.value))
def test_strict_spine_legacy_codec_matches_policy_schema(event_type: RuntimeEventType) -> None:
    policy = get_runtime_event_payload_policy(event_type)
    assert policy.schema_id is not None
    raw = dict(_REPRESENTATIVE_LEGACY_RAW.get(event_type, {}))
    typed, _promote = legacy_spine_payload_to_typed(event_type, raw)
    assert typed.schema_id == policy.schema_id
    validate_payload_envelope(typed.to_envelope())


def test_prepare_canonical_production_write_upgrades_decision_emitted() -> None:
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.DECISION_EMITTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload=_REPRESENTATIVE_LEGACY_RAW[RuntimeEventType.DECISION_EMITTED],
    )
    prepared = prepare_canonical_production_write_event(event)
    assert prepared.payload["payload_schema_id"] == "decision.v1"
    assert prepared.payload["data"]["decision_type"] == "continue"
    validate_payload_envelope(prepared.payload)


def test_unconvertible_decision_legacy_fails_closed() -> None:
    with pytest.raises(ValueError, match="decision_emitted payload missing"):
        legacy_spine_payload_to_typed(RuntimeEventType.DECISION_EMITTED, {"decision": "continue"})


def test_validating_persistence_rejects_unknown_schema_before_provider_append() -> None:
    identity = runtime_event_test_identity()
    store = ValidatingRuntimeEventPersistence(InMemoryRuntimeEventStore())
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "payload_schema_id": "unknown.schema.v99",
            "payload_schema_version": 1,
            "data": {},
        },
    )
    with pytest.raises(RuntimeEventSchemaError, match="unknown.schema.v99"):
        assert_runtime_event_schema(event)
    with pytest.raises(RuntimeEventSchemaError):
        store.append(event, tenant_id="tenant")


def test_domain_signal_extension_requires_typed_envelope() -> None:
    policy = get_runtime_event_payload_policy(RuntimeEventType.DOMAIN_SIGNAL)
    assert policy.classification == RuntimeEventTypeClassification.CANONICAL_PRODUCTION
    assert policy.write_mode == PayloadWriteMode.EXTENSION_EVENT_KIND
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        phase=ExecutionPhase.STEP_EXECUTION,
        event_kind="agents.custom.signal",
        payload={},
    )
    with pytest.raises(RuntimeEventSchemaError, match="extension event_kind"):
        assert_runtime_event_schema(event)


def test_ec1_write_missing_payload_schema_id_rejected() -> None:
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={},
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_write_wrong_payload_schema_id_rejected() -> None:
    event = _tool_requested_event(
        payload_schema_id="decision.v1",
        payload_schema_version=1,
        data={
            "decision_type": "continue",
            "reason": "policy",
        },
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_write_malformed_typed_payload_rejected() -> None:
    event = _tool_requested_event(
        payload_schema_id="tool.v1",
        payload_schema_version=1,
        data={"status": "requested"},
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_write_event_type_schema_mismatch_rejected() -> None:
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.AGENT_SELECTED,
        phase=ExecutionPhase.AGENT_SELECTION,
        payload={
            "payload_schema_id": "decision.v1",
            "payload_schema_version": 1,
            "data": {
                "decision_type": "continue",
                "reason": "policy",
            },
        },
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_canonical_spine_rejects_extension_schema_id() -> None:
    class _Ec1ExtensionOnly(RuntimeEventPayload):
        schema_id = "agents.ec1_diag.extension_only.v1"
        detail: str = ""

    register_extension_runtime_payload(_Ec1ExtensionOnly)
    event = _tool_requested_event(**_Ec1ExtensionOnly(detail="x").to_envelope())
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_extension_event_kind_without_envelope_rejected() -> None:
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        event_kind="agents.ec1_diag.untyped_signal",
        payload={},
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_invalid_extension_namespace_rejected() -> None:
    class _InvalidNamespacePayload(RuntimeEventPayload):
        schema_id = "vendor.custom.signal.v1"
        note: str = ""

    with pytest.raises(ExtensionSchemaError, match="runtime extension payload schema_id"):
        register_extension_runtime_payload(_InvalidNamespacePayload)
    with pytest.raises(DomainSignalError, match="event_kind must be"):
        validate_event_kind("Not_A_Valid_Kind")


def test_ec1_duplicate_schema_registration_rejected() -> None:
    class _Ec1DupA(RuntimeEventPayload):
        schema_id = "agents.ec1_diag.duplicate.v1"
        left: str = ""

    class _Ec1DupB(RuntimeEventPayload):
        schema_id = "agents.ec1_diag.duplicate.v1"
        right: str = ""

    register_extension_runtime_payload(_Ec1DupA)
    with pytest.raises(RuntimeEventPayloadError, match="duplicate payload schema_id"):
        register_extension_runtime_payload(_Ec1DupB)


def test_ec1_unconvertible_legacy_write_rejected_before_provider() -> None:
    event = RuntimeEvent(
        **runtime_event_test_identity(),
        event_type=RuntimeEventType.DECISION_EMITTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={"decision": "continue"},
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_invalid_bounded_observability_attribute_rejected_on_write() -> None:
    event = _tool_requested_event(
        payload_schema_id="tool.v1",
        payload_schema_version=1,
        data={
            "tool_name": "probe.tool",
            "status": "requested",
            "unsafe_vendor_blob": {"nested": "not-allowed"},
        },
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_historical_legacy_read_remains_compatible() -> None:
    legacy = {"tool_name": "legacy.tool", "status": "requested"}
    assert validate_payload_envelope(legacy) is None
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_tool_requested_event(**legacy))


def test_ec1_canonical_legacy_write_converts_or_rejects() -> None:
    inner = _AppendCountingStore()
    store = ValidatingRuntimeEventPersistence(inner)
    legacy_event = _tool_requested_event(
        **_REPRESENTATIVE_LEGACY_RAW[RuntimeEventType.TOOL_REQUESTED]
    )
    positioned = store.append(legacy_event, tenant_id="tenant-ec1")
    assert inner.append_calls == 1
    assert positioned.event.payload["payload_schema_id"] == "tool.v1"
    validate_payload_envelope(positioned.event.payload)

    _assert_invalid_canonical_write_rejects_before_provider(
        RuntimeEvent(
            **runtime_event_test_identity(),
            event_type=RuntimeEventType.AGENT_SELECTED,
            phase=ExecutionPhase.AGENT_SELECTION,
            payload={"selected_agent_id": ""},
        )
    )


def test_ec1_registered_extension_write_passes() -> None:
    clear_event_kind_registry()

    class _Ec1RegisteredSignal(RuntimeEventPayload):
        schema_id = "agents.ec1_diag.registered.v1"
        score: float = Field(ge=0.0, le=1.0)

    register_extension_runtime_payload(
        _Ec1RegisteredSignal,
        event_kind="agents.ec1_diag.registered",
    )
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        phase=ExecutionPhase.STEP_EXECUTION,
        event_kind="agents.ec1_diag.registered",
        payload=_Ec1RegisteredSignal(score=0.5).to_envelope(),
    )
    inner = _AppendCountingStore()
    store = ValidatingRuntimeEventPersistence(inner)
    positioned = store.append(event, tenant_id="tenant-ec1")
    assert inner.append_calls == 1
    validate_payload_envelope(positioned.event.payload)
    clear_event_kind_registry()


def test_ec1_domain_signal_does_not_open_generic_fallback_for_spine_events() -> None:
    identity = runtime_event_test_identity()
    extension_envelope = ToolPayloadV1(tool_name="x", status="requested").to_envelope()
    extension_envelope["payload_schema_id"] = "agents.ec1_diag.fallback_probe.v1"
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload=extension_envelope,
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_missing_payload_schema_envelope_string_rejected() -> None:
    with pytest.raises(RuntimeEventPayloadError, match="payload_schema_id must be"):
        validate_payload_envelope(
            {
                "payload_schema_id": "   ",
                "payload_schema_version": 1,
                "data": {},
            }
        )
    event = _tool_requested_event(
        payload_schema_id="   ",
        payload_schema_version=1,
        data={},
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)


def test_ec1_agent_selected_wrong_schema_data_shape_rejected() -> None:
    identity = runtime_event_test_identity()
    event = RuntimeEvent(
        **identity,
        event_type=RuntimeEventType.AGENT_SELECTED,
        phase=ExecutionPhase.AGENT_SELECTION,
        payload=AgentSelectionPayloadV1(
            selected_agent_id="agent.a",
        ).to_envelope()
        | {
            "payload_schema_id": "tool.v1",
        },
    )
    _assert_invalid_canonical_write_rejects_before_provider(event)
