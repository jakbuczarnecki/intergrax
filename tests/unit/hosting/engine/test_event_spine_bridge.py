# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting.eventing import (
    HOSTING_DOMAIN_EVENT_KIND,
    RuntimeSpineHostedApplicationEventPublisher,
    hosted_event_to_payload,
    register_hosting_domain_signal,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.events.emit_context import EmitContext

pytestmark = pytest.mark.unit


def test_payload_and_kind_registration_idempotent() -> None:
    register_hosting_domain_signal()
    register_hosting_domain_signal()


def test_runtime_spine_bridge_records_domain_signal() -> None:
    bus = RuntimeEventBus()
    publisher = RuntimeSpineHostedApplicationEventPublisher(bus)
    event = HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_READY,
        application_id="test_app",
        instance_id="instance-001",
        lifecycle_state=__import__("intergrax.hosting", fromlist=["HostedApplicationLifecycleState"]).HostedApplicationLifecycleState.READY,
        correlation_id="corr-1",
        causation_id="cause-1",
    )
    import asyncio

    asyncio.run(publisher.publish(event))
    recorded = bus.history
    assert len(recorded) == 1
    runtime_event = recorded[0]
    assert runtime_event.event_type is RuntimeEventType.DOMAIN_SIGNAL
    assert runtime_event.event_kind == HOSTING_DOMAIN_EVENT_KIND
    assert runtime_event.phase is ExecutionPhase.APPLICATION_HOSTING
    assert runtime_event.task_id == "hosting_test_app"
    assert runtime_event.run_id == "instance-001"
    payload = hosted_event_to_payload(event)
    assert payload.hosted_event_id == event.event_id


def test_emit_domain_signal_default_phase_unchanged() -> None:
    from intergrax.runtime.events.payloads.base import RuntimeEventPayload

    class _Payload(RuntimeEventPayload):
        schema_id = "intergrax.test.phase_default.v1"

        value: str = "ok"

        def redact(self) -> _Payload:
            return self

    from intergrax.runtime.events.payload_registry import register_payload_schema
    from intergrax.runtime.events.event_kind_registry import register_event_kind

    register_payload_schema(_Payload, extension=True)
    register_event_kind("applications.test.phase_default", _Payload.schema_id)
    bus = RuntimeEventBus()
    ctx = EmitContext(task_id="task-1", run_id="run-1", bus=bus)
    event = emit_domain_signal(ctx, kind="applications.test.phase_default", payload=_Payload())
    assert event.phase is ExecutionPhase.STEP_EXECUTION
