# © Artur Czarnecki. All rights reserved.

"""P1B-R3-R2 — deadline-before-work, health contract, worker terminal state."""

from __future__ import annotations

import inspect
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
    EventDeliveryObligation,
    EventDeliveryPolicy,
    EventDeliveryResult,
    EventPriority,
    EventSinkHealthPort,
    EventSinkHealthState,
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
from intergrax.runtime.observability.event_delivery.event_sink_health import MutableEventSinkHealth
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]

_PAYLOAD = make_observability_export_payload(event_id="ev-r2", kind="DECISION_FINALIZED")
_DELIVERABLE = make_deliverable_event(_PAYLOAD)


def _critical_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )


class _CustomHealthImplementation:
    """Contract-only health; not a subclass of MutableEventSinkHealth."""

    def __init__(self) -> None:
        self._state = EventSinkHealthState.HEALTHY

    def health_state(self) -> EventSinkHealthState:
        return self._state

    def mark_unhealthy(self) -> None:
        self._state = EventSinkHealthState.UNHEALTHY


def test_bounded_sink_constructor_uses_health_contract_only() -> None:
    sig = inspect.signature(BoundedEventSink.__init__)
    health_param = sig.parameters["health"]
    annotation = health_param.annotation
    assert "MutableEventSinkHealth" not in str(annotation)


def test_custom_health_implementation_via_protocol() -> None:
    health = _CustomHealthImplementation()

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

    assert isinstance(health, EventSinkHealthPort)
    policy = EventDeliveryPolicy(max_capacity=4)
    bounded = BoundedEventSink(_AcceptDownstream(), policy, health=health)
    health.mark_unhealthy()
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.REJECTED
    with pytest.raises(EventDeliveryBoundaryError):
        bounded.close()


def test_dead_worker_before_publish_marks_unhealthy_and_rejects() -> None:
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

    policy = EventDeliveryPolicy(max_capacity=4)
    bounded = BoundedEventSink(_AcceptDownstream(), policy)
    dead = threading.Thread()
    bounded._worker = dead  # noqa: SLF001 — test seam for unexpected worker death
    result = bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT)
    assert result.disposition is EventDeliveryDisposition.REJECTED
    assert result.disposition is not EventDeliveryDisposition.ACCEPTED
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY


def test_dead_worker_before_close_raises_and_marks_unhealthy() -> None:
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

    policy = EventDeliveryPolicy(max_capacity=4, drain_shutdown_timeout_seconds=1.0)
    bounded = BoundedEventSink(_AcceptDownstream(), policy)
    bounded._worker = threading.Thread()  # noqa: SLF001
    with pytest.raises(EventDeliveryBoundaryError) as raised:
        bounded.close()
    assert raised.value.kind is EventDeliveryBoundaryFailureKind.SINK_UNAVAILABLE
    assert bounded.health_state() is EventSinkHealthState.UNHEALTHY


def test_per_event_failure_keeps_worker_healthy() -> None:
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
    assert bounded.health_state() is EventSinkHealthState.HEALTHY
    assert bounded._worker.is_alive()  # noqa: SLF001
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    bounded.close()
    assert calls["count"] == 2


def test_critical_expired_deadline_skips_downstream_after_bus_timeout() -> None:
    hold_worker = threading.Event()
    release_holder = threading.Event()
    critical_downstream_calls: list[EventPriority] = []

    class _GatedDownstream:
        def publish(self, event, *, priority, deadline=None) -> EventDeliveryResult:
            if priority is EventPriority.BEST_EFFORT:
                hold_worker.set()
                release_holder.wait(timeout=3.0)
            else:
                critical_downstream_calls.append(priority)
            return EventDeliveryResult(
                disposition=EventDeliveryDisposition.ACCEPTED,
                priority=priority,
                buffered_depth=0,
                obligation=EventDeliveryObligation.COMPLETION
                if priority is EventPriority.CRITICAL
                else EventDeliveryObligation.ADMISSION,
            )

        def close(self) -> None:
            release_holder.set()

    policy = EventDeliveryPolicy(
        max_capacity=4,
        critical_completion_timeout_seconds=0.12,
        drain_shutdown_timeout_seconds=2.0,
    )
    bounded = BoundedEventSink(_GatedDownstream(), policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=0.12,
    )
    assert (
        bounded.publish(_DELIVERABLE, priority=EventPriority.BEST_EFFORT).disposition
        is EventDeliveryDisposition.ACCEPTED
    )
    assert hold_worker.wait(timeout=2.0)

    errors: list[BaseException] = []

    def _record_critical() -> None:
        try:
            bus.record(_critical_event())
        except CriticalEventDeliveryError as exc:
            errors.append(exc)

    t = threading.Thread(target=_record_critical)
    t.start()
    t.join(timeout=2.0)
    assert len(errors) == 1
    time.sleep(0.05)
    release_holder.set()
    time.sleep(0.1)
    assert critical_downstream_calls == []
    bounded.close()


def test_runtime_export_sink_past_deadline_does_not_call_export() -> None:
    called = threading.Event()

    class _Export:
        async def export(self, payload) -> None:
            called.set()

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    bridge = RuntimeEventExportSink(_Export())
    past = time.monotonic() - 1.0
    with pytest.raises(EventDeliveryBoundaryError) as raised:
        bridge.publish(_DELIVERABLE, priority=EventPriority.CRITICAL, deadline=past)
    assert raised.value.kind is EventDeliveryBoundaryFailureKind.COMPLETION_TIMEOUT
    assert not called.is_set()


def test_runtime_export_sink_remaining_deadline_propagated() -> None:
    from unittest.mock import patch

    class _Export:
        async def export(self, payload) -> None:
            return None

        async def flush(self) -> None:
            return None

        async def close(self) -> None:
            return None

    bridge = RuntimeEventExportSink(_Export())
    future_deadline = time.monotonic() + 0.5
    with patch.object(
        bridge._runner,
        "run",
        wraps=bridge._runner.run,
    ) as mock_run:
        result = bridge.publish(
            _DELIVERABLE,
            priority=EventPriority.CRITICAL,
            deadline=future_deadline,
        )
    assert result.disposition is EventDeliveryDisposition.ACCEPTED
    timeout_seconds = mock_run.call_args.kwargs["timeout_seconds"]
    assert 0.0 < timeout_seconds <= 0.5


def test_critical_real_stack_regression() -> None:
    policy = EventDeliveryPolicy(
        max_capacity=8,
        critical_completion_timeout_seconds=2.0,
        drain_shutdown_timeout_seconds=2.0,
    )
    recorder = RecordingEventExportSink()
    bridge = RuntimeEventExportSink(recorder)
    bounded = BoundedEventSink(bridge, policy)
    bus = RuntimeEventBus(
        record_history=False,
        event_sink=bounded,
        critical_completion_timeout_seconds=2.0,
    )
    bus.record(_critical_event())
    assert len(recorder.payloads) == 1
    bounded.close()


def test_architecture_gate_r2_no_concrete_coupling() -> None:
    bounded_src = (
        _REPO / "intergrax/runtime/observability/event_delivery/bounded_event_sink.py"
    ).read_text(encoding="utf-8")
    bus_src = (_REPO / "intergrax/runtime/events/event_bus.py").read_text(encoding="utf-8")
    sig = inspect.signature(BoundedEventSink.__init__)
    assert sig.parameters["health"].annotation is not inspect.Parameter.empty
    assert "MutableEventSinkHealth" not in str(sig.parameters["health"].annotation)
    assert "RuntimeEventExportSink" not in bounded_src
    assert "BoundedEventSink" not in bus_src
    assert "isinstance" not in bounded_src
