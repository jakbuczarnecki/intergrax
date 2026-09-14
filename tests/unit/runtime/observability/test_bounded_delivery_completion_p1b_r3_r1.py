# © Artur Czarnecki. All rights reserved.

"""P1B-R3-R1 — bounded delivery completion semantics."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from intergrax.contracts.event_delivery import (
    CriticalEventDeliveryError,
    DeliverableEvent,
    EventDeliveryBoundaryError,
    EventDeliveryBoundaryFailureKind,
    EventDeliveryDisposition,
    EventDeliveryLateFailure,
    EventDeliveryObligation,
    EventDeliveryObligationPolicyPort,
    EventDeliveryPolicy,
    EventDeliveryPostAdmissionFailureObserverPort,
    EventDeliveryResult,
    EventPriority,
    effective_event_delivery_obligation,
    make_deliverable_event,
    make_observability_export_payload,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    RecordingEventExportSink,
    RuntimeEventExportSink,
)
from intergrax.runtime.observability.event_delivery.enterprise_default_event_delivery_obligation_policy import (
    EnterpriseDefaultEventDeliveryObligationPolicy,
)
from intergrax.runtime.observability.event_delivery.event_sink_health import MutableEventSinkHealth
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]

_PAYLOAD = make_observability_export_payload(event_id="ev-1", kind="DECISION_FINALIZED")
_DELIVERABLE = make_deliverable_event(_PAYLOAD)


def _critical_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )


def _best_effort_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        **runtime_event_test_identity(),
    )


def test_contract_obligation_semantics_and_critical_floor() -> None:
    policy = EnterpriseDefaultEventDeliveryObligationPolicy()
    assert policy.obligation_for(EventPriority.BEST_EFFORT) is EventDeliveryObligation.ADMISSION
    assert policy.obligation_for(EventPriority.IMPORTANT) is EventDeliveryObligation.ADMISSION
    assert policy.obligation_for(EventPriority.CRITICAL) is EventDeliveryObligation.COMPLETION

    class _AdmissionOnlyCritical:
        def obligation_for(self, priority: EventPriority) -> EventDeliveryObligation:
            return EventDeliveryObligation.ADMISSION

    assert (
        effective_event_delivery_obligation(EventPriority.CRITICAL, _AdmissionOnlyCritical())
        is EventDeliveryObligation.COMPLETION
    )


def test_admission_returns_before_slow_downstream() -> None:
    gate = threading.Event()

    class _SlowDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            gate.wait(timeout=2.0)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            gate.set()

    policy = EventDeliveryPolicy(max_capacity=8, important_wait_timeout_seconds=0.05)
    bounded = BoundedEventSink(_SlowDownstream(), policy)
    started = time.monotonic()
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    elapsed = time.monotonic() - started
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    assert result.obligation is EventDeliveryObligation.ADMISSION
    assert elapsed < 0.3
    gate.set()
    bounded.close()


def _bounded_export_stack(
    *,
    critical_timeout: float = 2.0,
    policy: EventDeliveryPolicy | None = None,
) -> tuple[RuntimeEventBus, BoundedEventSink, RecordingEventExportSink]:
    resolved_policy = policy or EventDeliveryPolicy(
        max_capacity=8,
        critical_completion_timeout_seconds=critical_timeout,
        drain_shutdown_timeout_seconds=2.0,
    )
    recorder = RecordingEventExportSink()
    bridge = RuntimeEventExportSink(recorder)
    bounded = BoundedEventSink(bridge, resolved_policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=critical_timeout,
    )
    return bus, bounded, recorder


def test_critical_completion_success_real_stack() -> None:
    bus, bounded, recorder = _bounded_export_stack()
    bus.record(_critical_event())
    assert len(recorder.payloads) == 1
    bounded.close()


def test_critical_downstream_reject_fails_bus() -> None:
    class _RejectDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            return None

    policy = EventDeliveryPolicy(max_capacity=4, critical_completion_timeout_seconds=2.0)
    bounded = BoundedEventSink(_RejectDownstream(), policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=2.0,
    )
    with pytest.raises(CriticalEventDeliveryError):
        bus.record(_critical_event())
    bounded.close()


def test_critical_boundary_error_preserves_cause() -> None:
    boundary = EventDeliveryBoundaryError(
        kind=EventDeliveryBoundaryFailureKind.TRANSPORT_FAILURE,
        message="downstream broke",
        deliverable_event_id=_DELIVERABLE.event_id,
    )

    class _BoundaryDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            raise boundary

        def close(self) -> None:
            return None

    policy = EventDeliveryPolicy(max_capacity=4, critical_completion_timeout_seconds=2.0)
    bounded = BoundedEventSink(_BoundaryDownstream(), policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=2.0,
    )
    with pytest.raises(CriticalEventDeliveryError) as raised:
        bus.record(_critical_event())
    assert raised.value.__cause__ is boundary
    bounded.close()


def test_critical_completion_timeout() -> None:
    gate = threading.Event()

    class _HangingExport:
        async def export(self, payload) -> None:
            gate.wait(timeout=5.0)

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            gate.set()

    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_completion_timeout_seconds=0.15,
        drain_shutdown_timeout_seconds=2.0,
    )
    bridge = RuntimeEventExportSink(_HangingExport())
    bounded = BoundedEventSink(bridge, policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=0.15,
    )
    with pytest.raises(CriticalEventDeliveryError):
        bus.record(_critical_event())
    gate.set()
    bounded.close()


def test_worker_survives_per_event_failure() -> None:
    calls = {"count": 0}

    class _FlakyDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            calls["count"] += 1
            if calls["count"] == 1:
                return EventDeliveryResult(
                    disposition=EventDeliveryDisposition.REJECTED,
                    priority=priority,
                    buffered_depth=0,
                    obligation=EventDeliveryObligation.ADMISSION,
                )
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            return None

    policy = EventDeliveryPolicy(max_capacity=8)
    bounded = BoundedEventSink(_FlakyDownstream(), policy)
    bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    time.sleep(0.05)
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    bounded.close()
    assert calls["count"] == 2


def test_late_failure_observer_for_admission() -> None:
    captured: list[EventDeliveryLateFailure] = []

    class _Observer:
        def on_late_failure(self, failure: EventDeliveryLateFailure) -> None:
            captured.append(failure)

    class _FailingDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.REJECTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            return None

    policy = EventDeliveryPolicy(max_capacity=8)
    bounded = BoundedEventSink(
        _FailingDownstream(),
        policy,
        late_failure_observer=_Observer(),
    )
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    time.sleep(0.1)
    bounded.close()
    assert len(captured) == 1
    assert captured[0].disposition is EventDeliveryDisposition.REJECTED


def test_unhealthy_sink_rejects_publish() -> None:
    health = MutableEventSinkHealth()
    policy = EventDeliveryPolicy(max_capacity=4)

    class _AcceptDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            return None

    bounded = BoundedEventSink(_AcceptDownstream(), policy, health=health)
    health.mark_unhealthy()
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.REJECTED
    bounded.close()


def test_unexpected_plugin_defect_normalized_to_boundary() -> None:
    class _BrokenExport:
        async def export(self, payload) -> None:
            raise ValueError("plugin defect")

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    bridge = RuntimeEventExportSink(_BrokenExport())
    with pytest.raises(EventDeliveryBoundaryError) as raised:
        bridge.publish(_DELIVERABLE, priority=EventPriority.IMPORTANT)
    assert raised.value.kind is EventDeliveryBoundaryFailureKind.INTERNAL_ERROR
    assert isinstance(raised.value.__cause__, ValueError)


def test_plugin_custom_obligation_policy_injection() -> None:
    class _ImportantCompletionPolicy:
        def obligation_for(self, priority: EventPriority) -> EventDeliveryObligation:
            if priority is EventPriority.IMPORTANT:
                return EventDeliveryObligation.COMPLETION
            return EventDeliveryObligation.ADMISSION

    gate = threading.Event()

    class _SlowDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            gate.wait(timeout=2.0)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION,
            )

        def close(self) -> None:
            gate.set()

    policy = EventDeliveryPolicy(max_capacity=4, critical_completion_timeout_seconds=1.0)
    bounded = BoundedEventSink(
        _SlowDownstream(),
        policy,
        obligation_policy=_ImportantCompletionPolicy(),
    )

    def _release() -> None:
        time.sleep(0.05)
        gate.set()

    threading.Thread(target=_release, daemon=True).start()
    started = time.monotonic()
    result = bounded.publish(
        _DELIVERABLE,
        priority=EventPriority.IMPORTANT,
        deadline=time.monotonic() + 1.0,
    )
    assert time.monotonic() - started >= 0.04
    assert result.obligation is EventDeliveryObligation.COMPLETION
    bounded.close()


def test_shutdown_drains_and_terminates_worker() -> None:
    bus, bounded, recorder = _bounded_export_stack()
    bus.record(_best_effort_event())
    time.sleep(0.05)
    bounded.close()
    assert bounded.closed
    assert len(recorder.payloads) == 1


def test_architecture_gate_no_concrete_coupling() -> None:
    bounded_src = (
        _REPO / "intergrax/runtime/observability/event_delivery/bounded_event_sink.py"
    ).read_text(encoding="utf-8")
    bus_src = (_REPO / "intergrax/runtime/events/event_bus.py").read_text(encoding="utf-8")
    assert "RuntimeEventExportSink" not in bounded_src
    assert "BoundedEventSink" not in bus_src
    assert "isinstance" not in bounded_src
