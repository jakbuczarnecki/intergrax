# © Artur Czarnecki. All rights reserved.

"""W5-D — observability exporter composition boundary qualification."""

from __future__ import annotations

import asyncio
import time

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.event_delivery import EventDeliveryPolicy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.observability_export import (
    ExportError,
    ExporterKind,
    ObservabilityExportProfile,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    NoopEventExportSink,
    OtlpEventExportSink,
    ObservabilityExportSinkFactory,
    RecordingEventExportSink,
    RuntimeEventExportSink,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _bounded_env(
    *,
    profile_id: str = "w5.d",
    exporter_kind: ExporterKind = ExporterKind.NOOP,
) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": exporter_kind,
        },
    )
    return env


def _event(kind_suffix: str = "") -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind=f"qualification.w5d{kind_suffix}",
        **runtime_event_test_identity(),
    )


@pytest.mark.parametrize(
    ("profile", "expected_type"),
    [
        (
            ObservabilityExportProfile(enabled=False, exporter_kind=ExporterKind.NOOP),
            NoopEventExportSink,
        ),
        (
            ObservabilityExportProfile(enabled=True, exporter_kind=ExporterKind.RECORDING),
            RecordingEventExportSink,
        ),
        (
            ObservabilityExportProfile(enabled=True, exporter_kind=ExporterKind.OTLP),
            OtlpEventExportSink,
        ),
    ],
)
def test_factory_selection(profile: ObservabilityExportProfile, expected_type: type) -> None:
    factory = ObservabilityExportSinkFactory()
    sink = factory.create(profile)
    assert isinstance(sink, expected_type)


def test_factory_instances_are_not_singletons() -> None:
    factory_a = ObservabilityExportSinkFactory()
    factory_b = ObservabilityExportSinkFactory()
    assert factory_a is not factory_b
    profile = ObservabilityExportProfile(enabled=True, exporter_kind=ExporterKind.RECORDING)
    sink_a = factory_a.create(profile)
    sink_b = factory_b.create(profile)
    assert sink_a is not sink_b


def test_ten_runtime_instances_isolated_exporters() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    sinks: list[object] = []
    runtimes = []
    for index in range(10):
        env = _bounded_env(profile_id=f"w5.d.iso.{index}", exporter_kind=ExporterKind.RECORDING)
        runtime = build_harness_host_runtime(manifest, env, settings=settings)
        runtimes.append(runtime)
        delivery = runtime.env_wiring.event_delivery
        assert delivery.event_export_sink is not None
        sinks.append(delivery.event_export_sink)
    assert len({id(s) for s in sinks}) == 10
    for runtime in runtimes:
        runtime.close()


class _ExportErrorSink:
    async def export(self, event: RuntimeEvent) -> None:
        raise ExportError("export failed")

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_export_failure_isolation_with_export_error() -> None:
    metrics = InternalDeliveryMetrics(exporter_kind="recording")
    bridge = RuntimeEventExportSink(_ExportErrorSink(), delivery_metrics=metrics)
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=16))
    bus = RuntimeEventBus(event_sink=bounded, delivery_metrics=metrics)
    handler_calls = 0

    def _handler(_event: RuntimeEvent) -> None:
        nonlocal handler_calls
        handler_calls += 1

    bus.subscribe(_handler, event_types={RuntimeEventType.TASK_PROGRESS})
    await bus.publish(_event(".export_error"))
    await asyncio.sleep(0.05)
    assert handler_calls == 1
    snap = metrics.snapshot()
    assert snap.export_attempt_total >= 1
    assert snap.export_failed_total >= 1
    bus.close()


class _OrderTrackingSink:
    def __init__(self) -> None:
        self.steps: list[str] = []

    async def export(self, event: RuntimeEvent) -> None:
        return None

    async def flush(self) -> None:
        self.steps.append("flush")

    async def close(self) -> None:
        self.steps.append("close")


def test_shutdown_flush_before_exporter_close() -> None:
    sink = _OrderTrackingSink()
    bridge = RuntimeEventExportSink(sink)
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=8))
    bus = RuntimeEventBus(event_sink=bounded)
    bus.close()
    assert sink.steps == ["flush", "close"]


class _SlowExportSink:
    async def export(self, event: RuntimeEvent) -> None:
        time.sleep(0.05)

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_backpressure_owned_by_bounded_sink_not_exporter() -> None:
    bridge = RuntimeEventExportSink(_SlowExportSink())
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=4))
    assert isinstance(bounded, BoundedEventSink)
    assert bounded._queue.maxsize == 4  # noqa: SLF001
    bus = RuntimeEventBus(event_sink=bounded)
    for index in range(20):
        await bus.publish(_event(f".bp.{index}"))
    await asyncio.sleep(0.3)
    bus.close()


def test_wiring_recording_profile_from_environment() -> None:
    env = _bounded_env(exporter_kind=ExporterKind.RECORDING)
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.export_profile is not None
    assert wiring.export_profile.exporter_kind is ExporterKind.RECORDING
    assert isinstance(wiring.event_export_sink, RecordingEventExportSink)
    snap = wiring.delivery_metrics.snapshot() if wiring.delivery_metrics else None
    assert snap is not None
    assert snap.exporter_kind == "recording"
