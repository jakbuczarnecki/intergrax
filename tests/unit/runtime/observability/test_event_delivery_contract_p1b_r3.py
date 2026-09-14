# © Artur Czarnecki. All rights reserved.

"""P1B-R3 — contract-pure runtime event delivery boundary."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    DeliverableEvent,
    EventDeliveryBoundaryError,
    EventDeliveryBoundaryFailureKind,
    EventDeliveryDisposition,
    EventDeliveryPolicy,
    EventDeliveryReaction,
    EventDeliveryResult,
    EventExportSinkPort,
    EventPriority,
    EventSinkPort,
    make_deliverable_event,
    make_observability_export_payload,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InMemoryEventSink,
    RecordingEventExportSink,
    RuntimeEventExportSink,
    runtime_event_to_deliverable,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_sink_delivery_reaction import (
    EnterpriseDefaultEventSinkDeliveryReaction,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def test_deliverable_event_single_identity_storage() -> None:
    payload = make_observability_export_payload(event_id="e-1", kind="k-1")
    deliverable = make_deliverable_event(payload, sequence=3)
    assert deliverable.event_id == "e-1"
    assert deliverable.kind == "k-1"
    names = {field.name for field in fields(DeliverableEvent)}
    assert names == {"export_payload", "sequence"}


def test_runtime_mapper_builds_payload_not_duplicate_fields() -> None:
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind="qualification.p1b-r3",
        **runtime_event_test_identity(),
    )
    deliverable = runtime_event_to_deliverable(event)
    assert deliverable.export_payload.event_id == str(event.event_id)
    assert deliverable.kind == event.event_kind


def test_event_export_sink_port_rejects_object_signature() -> None:
    sig = inspect.signature(EventExportSinkPort.export)
    assert sig.parameters["payload"].annotation is not object


def test_architecture_gate_no_delivery_path_coupling() -> None:
    bounded_src = (_REPO / "intergrax/runtime/observability/event_delivery/bounded_event_sink.py").read_text(
        encoding="utf-8",
    )
    bus_src = (_REPO / "intergrax/runtime/events/event_bus.py").read_text(encoding="utf-8")
    export_src = (
        _REPO / "intergrax/runtime/observability/event_delivery/runtime_event_export_sink.py"
    ).read_text(encoding="utf-8")
    assert "source_event" not in bounded_src
    assert "deliver_bounded" not in export_src
    assert "BoundedEventSink" not in bus_src
    assert "RuntimeEventExportSink" not in bounded_src
    assert "isinstance" not in bounded_src


class _ContinueAlwaysReaction:
    def react_to_result(self, *, priority, result, deliverable) -> EventDeliveryReaction:
        _ = (priority, result, deliverable)
        return EventDeliveryReaction.CONTINUE

    def react_to_boundary_error(self, *, priority, error, deliverable) -> EventDeliveryReaction:
        _ = (priority, error, deliverable)
        return EventDeliveryReaction.CONTINUE


class _RejectingSink:
    def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
        _ = (event, deadline)
        return EventDeliveryResult(
            disposition=EventDeliveryDisposition.REJECTED,
            priority=priority,
            buffered_depth=0,
        )

    def close(self) -> None:
        return None


def test_bus_critical_floor_cannot_be_weakened_by_custom_reaction() -> None:
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=_RejectingSink(),
        delivery_reaction=_ContinueAlwaysReaction(),
    )
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )
    with pytest.raises(CriticalEventDeliveryError):
        bus.record(event)


def test_bus_best_effort_failure_default_continue() -> None:
    bus = RuntimeEventBus(record_history=False, event_sink=_RejectingSink())
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        **runtime_event_test_identity(),
    )
    bus.record(event)


class _BoundaryErrorSink:
    def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
        _ = (event, priority, deadline)
        raise EventDeliveryBoundaryError(
            kind=EventDeliveryBoundaryFailureKind.TRANSPORT_FAILURE,
            message="simulated",
            deliverable_event_id=event.event_id,
        )

    def close(self) -> None:
        return None


def test_bus_critical_boundary_error_fails_execution() -> None:
    bus = RuntimeEventBus(record_history=False, event_sink=_BoundaryErrorSink())
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )
    with pytest.raises(CriticalEventDeliveryError):
        bus.record(event)


def test_custom_export_sink_accepts_observability_export_payload() -> None:
    recorder = RecordingEventExportSink()
    bridge = RuntimeEventExportSink(recorder)
    payload = make_observability_export_payload(event_id="x", kind="y")
    deliverable = make_deliverable_event(payload)
    result = bridge.publish(deliverable, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    assert recorder.payloads[0].kind == "y"


def test_enterprise_default_reaction_matches_critical_reject() -> None:
    reaction = EnterpriseDefaultEventSinkDeliveryReaction()
    deliverable = make_deliverable_event(make_observability_export_payload(event_id="a", kind="b"))
    result = EventDeliveryResult(
        disposition=EventDeliveryDisposition.REJECTED,
        priority=EventPriority.CRITICAL,
        buffered_depth=0,
    )
    assert (
        reaction.react_to_result(
            priority=EventPriority.CRITICAL,
            result=result,
            deliverable=deliverable,
        )
        is EventDeliveryReaction.FAIL_EXECUTION
    )


def test_plugin_custom_event_sink_port_without_core_changes() -> None:
    captured: list[DeliverableEvent] = []

    class _CustomSink:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            _ = (priority, deadline)
            captured.append(event)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=EventPriority.BEST_EFFORT,
                buffered_depth=1,
            )

        def close(self) -> None:
            return None

    sink: EventSinkPort = _CustomSink()
    bounded = BoundedEventSink(sink, EventDeliveryPolicy(max_capacity=4))
    bus = RuntimeEventBus(record_history=False, event_sink=bounded)
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )
    bus.record(event)
    bounded.close()
    assert len(captured) == 1
