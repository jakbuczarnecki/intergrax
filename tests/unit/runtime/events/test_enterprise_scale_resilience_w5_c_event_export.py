# © Artur Czarnecki. All rights reserved.

"""W5-C — pluggable downstream event export sink pipeline qualification."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.contracts.event_delivery import EventDeliveryPolicy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    RecordingEventExportSink,
    RuntimeEventExportSink,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _bounded_env(profile_id: str = "w5.c") -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.observability_profile = env.observability_profile.model_copy(
        update={"bounded_event_delivery_enabled": True},
    )
    return env


def _event(kind_suffix: str = "") -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind=f"qualification.w5c{kind_suffix}",
        **runtime_event_test_identity(),
    )


def _wired_bus_with_recorder() -> tuple[RuntimeEventBus, RecordingEventExportSink, BoundedEventSink]:
    metrics = InternalDeliveryMetrics()
    recorder = RecordingEventExportSink()
    bridge = RuntimeEventExportSink(recorder, delivery_metrics=metrics)
    bounded = BoundedEventSink(
        bridge,
        EventDeliveryPolicy(max_capacity=128),
    )
    bus = RuntimeEventBus(event_sink=bounded, delivery_metrics=metrics)
    return bus, recorder, bounded


@pytest.mark.asyncio
async def test_routing_exporter_receives_published_event() -> None:
    bus, recorder, bounded = _wired_bus_with_recorder()
    event = _event(".routing")
    await bus.publish(event)
    await asyncio.sleep(0.05)
    bus.close()
    assert len(recorder.events) == 1
    assert recorder.events[0].event_kind == event.event_kind
    assert bounded.closed


@pytest.mark.asyncio
async def test_ordering_preserved_for_100_events() -> None:
    bus, recorder, bounded = _wired_bus_with_recorder()
    for index in range(1, 101):
        await bus.publish(_event(f".event_{index}"))
    await asyncio.sleep(0.2)
    bus.close()
    kinds = [e.event_kind for e in recorder.events]
    assert len(kinds) == 100
    for index in range(1, 101):
        assert kinds[index - 1] == f"qualification.w5c.event_{index}"
    assert bounded.closed


class _FailingExportSink:
    async def export(self, event: RuntimeEvent) -> None:
        raise RuntimeError("export failed")

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_export_failure_does_not_break_execution() -> None:
    metrics = InternalDeliveryMetrics()
    bridge = RuntimeEventExportSink(_FailingExportSink(), delivery_metrics=metrics)
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=16))
    bus = RuntimeEventBus(event_sink=bounded, delivery_metrics=metrics)
    handler_calls = 0

    def _handler(_event: RuntimeEvent) -> None:
        nonlocal handler_calls
        handler_calls += 1

    bus.subscribe(_handler, event_types={RuntimeEventType.TASK_PROGRESS})
    await bus.publish(_event(".failure"))
    await asyncio.sleep(0.05)
    assert handler_calls == 1
    snap = metrics.snapshot()
    assert snap.export_failed >= 1
    bus.close()


def test_shutdown_flush_and_close_exporter() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    runtime = build_harness_host_runtime(manifest, _bounded_env(), settings=settings)
    delivery = runtime.env_wiring.event_delivery
    bridge = delivery.export_bridge
    assert bridge is not None
    runtime.close()
    assert bridge.closed
    bus = runtime.env_wiring.build_context.runtime_event_bus
    assert bus is not None
    assert bus.closed


def test_instance_isolation_between_environments() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    runtime_a = build_harness_host_runtime(
        manifest,
        _bounded_env("w5.c.a"),
        settings=settings,
    )
    runtime_b = build_harness_host_runtime(
        manifest,
        _bounded_env("w5.c.b"),
        settings=settings,
    )
    export_a = runtime_a.env_wiring.event_delivery.event_export_sink
    export_b = runtime_b.env_wiring.event_delivery.event_export_sink
    assert export_a is not None
    assert export_b is not None
    assert export_a is not export_b
    runtime_a.close()
    runtime_b.close()


def test_legacy_bus_without_exporter_unchanged() -> None:
    bus = RuntimeEventBus()
    assert bus.event_sink is None
    assert bus.delivery_metrics is None
    bus.close()
    assert bus.closed
