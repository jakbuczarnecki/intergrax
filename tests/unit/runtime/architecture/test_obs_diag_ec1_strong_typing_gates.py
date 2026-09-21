# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-EC1 — strong typing gates for canonical OBS integration contracts."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest

from intergrax.contracts.execution_identity import EventId, mint_event_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.integrations.contracts.observability_backend import TraceRecord
from intergrax.runtime.events.evidence_persistence_adapter import as_evidence_persistence_port
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.execution_position import ExecutionEventPosition, PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError
from intergrax.runtime.events.spine_payload_codec import prepare_canonical_production_write_event
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.validating_evidence_persistence_port import (
    CanonicalRuntimeEventWriteValidatedPort,
    ValidatingEvidencePersistencePort,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OBS_BACKEND_CONTRACT = _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "observability_backend.py"

_LEGACY_DECISION_PAYLOAD = {
    "decision": "continue",
    "reason": "policy allowed",
    "step_id": "step-1",
    "policy_action": "allow",
}


def _legacy_decision_emitted_event() -> RuntimeEvent:
    identity = runtime_event_test_identity()
    return RuntimeEvent(
        **identity,
        event_id=mint_event_id(),
        event_type=RuntimeEventType.DECISION_EMITTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload=dict(_LEGACY_DECISION_PAYLOAD),
    )


class _AppendCapturingEvidencePort:
    """Minimal ``EvidencePersistencePort`` that records durable append inputs."""

    def __init__(self) -> None:
        self.append_calls = 0
        self.append_inputs: list[RuntimeEvent] = []

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        self.append_calls += 1
        self.append_inputs.append(event)
        return PositionedRuntimeEvent(event=event, position=ExecutionEventPosition(1))

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
        after: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        return []

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return []

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return TaskRuntimeEventRuns(runs=())

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        return None

    def list_positioned_through(
        self,
        boundary: object,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[PositionedRuntimeEvent]:
        return []


def test_trace_record_metadata_is_typed_observability_attributes() -> None:
    from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue

    field = TraceRecord.model_fields["metadata"]
    assert field.annotation == dict[str, ObservabilityAttributeValue]


def test_trace_record_nested_arbitrary_object_not_semantic_metadata() -> None:
    record = TraceRecord(
        trace_id="trace-ec1",
        name="gate",
        metadata={
            "safe_key": "scalar",
            "nested_arbitrary": {"child": object()},
        },
    )
    assert record.metadata == {"safe_key": "scalar"}
    assert "nested_arbitrary" not in record.metadata


def test_observability_backend_contract_no_raw_any_metadata_annotation() -> None:
    tree = ast.parse(_OBS_BACKEND_CONTRACT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "TraceRecord":
            continue
        for item in node.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id == "metadata"
                and item.annotation is not None
            ):
                src = ast.unparse(item.annotation)
                assert "Any" not in src
                return
    pytest.fail("TraceRecord.metadata annotation not found")


def test_interrupt_payload_metadata_is_typed_observability_attributes() -> None:
    from intergrax.contracts.application_observability_attributes import ObservabilityAttributeValue
    from intergrax.runtime.events.payloads.canonical import InterruptPayloadV1

    field = InterruptPayloadV1.model_fields["metadata"]
    assert field.annotation == dict[str, ObservabilityAttributeValue]


def test_runtime_event_payload_policy_covers_all_enum_members() -> None:
    from intergrax.runtime.events.runtime_event_payload_policy import iter_runtime_event_payload_policies

    covered = {event_type for event_type, _ in iter_runtime_event_payload_policies()}
    assert covered == set(RuntimeEventType)


def test_as_evidence_persistence_port_wraps_custom_port_and_prepares_canonical_append() -> None:
    inner = _AppendCapturingEvidencePort()
    port = as_evidence_persistence_port(inner)
    assert isinstance(port, ValidatingEvidencePersistencePort)
    assert port.inner is inner

    raw = _legacy_decision_emitted_event()
    positioned = port.append(raw, tenant_id="tenant-ec1")
    assert inner.append_calls == 1
    assert inner.append_inputs[0].payload != raw.payload
    assert inner.append_inputs[0].payload["payload_schema_id"] == "decision.v1"
    assert positioned.event.payload == inner.append_inputs[0].payload


def test_as_evidence_persistence_port_rejects_invalid_canonical_before_inner_append() -> None:
    inner = _AppendCapturingEvidencePort()
    port = as_evidence_persistence_port(inner)
    identity = runtime_event_test_identity()
    invalid = RuntimeEvent(
        **identity,
        event_id=mint_event_id(),
        event_type=RuntimeEventType.TOOL_REQUESTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "payload_schema_id": "unknown.schema.v99",
            "payload_schema_version": 1,
            "data": {},
        },
    )
    with pytest.raises(RuntimeEventSchemaError):
        port.append(invalid, tenant_id="tenant-ec1")
    assert inner.append_calls == 0


def test_as_evidence_persistence_port_does_not_double_wrap_validated_port() -> None:
    inner = _AppendCapturingEvidencePort()
    validated = ValidatingEvidencePersistencePort(inner)
    assert isinstance(validated, CanonicalRuntimeEventWriteValidatedPort)
    wrapped = as_evidence_persistence_port(validated)
    assert wrapped is validated


def test_event_bus_history_and_handlers_receive_committed_canonical_representation() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    handler_events: list[RuntimeEvent] = []
    bus.subscribe(
        handler_events.append,
        event_types={RuntimeEventType.DECISION_EMITTED},
    )

    raw = _legacy_decision_emitted_event()
    bus.record(raw, tenant_id="tenant-ec1")

    assert raw.payload == _LEGACY_DECISION_PAYLOAD
    assert "payload_schema_id" not in raw.payload
    committed_history = bus.history[-1]
    assert committed_history.payload["payload_schema_id"] == "decision.v1"
    assert len(handler_events) == 1
    handler_event = handler_events[0]
    assert handler_event.payload == committed_history.payload
    assert handler_event.payload != raw.payload

    positioned_rows = store.list_positioned_for_run(
        str(raw.run_id),
        tenant_id="tenant-ec1",
    )
    assert len(positioned_rows) == 1
    persisted = positioned_rows[0].event
    assert persisted.payload == committed_history.payload
    assert persisted.payload == handler_event.payload
    expected = prepare_canonical_production_write_event(raw)
    assert committed_history.payload == expected.payload


def test_validating_evidence_port_has_no_reflection_lifecycle() -> None:
    path = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "events"
        / "validating_evidence_persistence_port.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "def close" not in source


def test_evidence_persistence_port_protocol_has_no_lifecycle() -> None:
    path = _REPO_ROOT / "intergrax" / "contracts" / "execution_evidence" / "persistence_port.py"
    source = path.read_text(encoding="utf-8")
    assert "def close" not in source
