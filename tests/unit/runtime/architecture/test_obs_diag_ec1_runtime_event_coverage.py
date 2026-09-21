# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1 — strict-write codec coverage and negative write matrix."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.runtime_event_payload_policy import (
    CANONICAL_PRODUCTION_STRICT_SPINE_EVENT_TYPES,
    PayloadWriteMode,
    RuntimeEventTypeClassification,
    get_runtime_event_payload_policy,
    iter_runtime_event_payload_policies,
)
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.runtime.events.spine_payload_codec import (
    legacy_spine_payload_to_typed,
    prepare_canonical_production_write_event,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_EXEC_ID = mint_execution_id()

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
