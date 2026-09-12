# © Artur Czarnecki. All rights reserved.

"""W5-F — distributed observability transport boundary qualification."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry.exporter.otlp.proto.http._log_exporter")

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.event_delivery import EventDeliveryPolicy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.observability_export import (
    ConfigurationError,
    ExporterKind,
    ObservabilityExportProfile,
    OtlpProtocol,
    OtlpTransportPort,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    OtlpEventExportSink,
    ObservabilityExportSinkFactory,
    RuntimeEventExportSink,
)
from intergrax.runtime.observability.exporters.distributed.collector_transport import (
    CollectorTransport,
)
from intergrax.runtime.observability.exporters.distributed.distributed_configuration import (
    DistributedTransportConfiguration,
    validate_distributed_transport_configuration,
)
from intergrax.runtime.observability.exporters.distributed.errors import (
    DistributedTransportError,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTOR_ENDPOINT = "http://127.0.0.1:4318/v1/logs"
_SERVICE_NAME = "intergrax.distributed.qualification"


def _event(kind_suffix: str = "") -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind=f"qualification.w5f{kind_suffix}",
        **runtime_event_test_identity(),
    )


def _distributed_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.DISTRIBUTED_OTLP,
            "otlp_export_endpoint": _COLLECTOR_ENDPOINT,
            "observability_export_service_name": _SERVICE_NAME,
        },
    )
    return env


def _local_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.OTLP,
            "otlp_export_endpoint": _COLLECTOR_ENDPOINT,
        },
    )
    return env


def test_factory_selection_local_vs_distributed_transport() -> None:
    local_wiring = resolve_application_runtime_event_delivery_wiring(_local_env("w5.f.local"))
    distributed_wiring = resolve_application_runtime_event_delivery_wiring(
        _distributed_env("w5.f.distributed"),
    )
    assert local_wiring.otlp_transport is not None
    assert distributed_wiring.otlp_transport is not None
    assert isinstance(local_wiring.otlp_transport, OtlpTransport)
    assert isinstance(distributed_wiring.otlp_transport, CollectorTransport)
    assert isinstance(local_wiring.event_export_sink, OtlpEventExportSink)
    assert isinstance(distributed_wiring.event_export_sink, OtlpEventExportSink)


def test_ten_wiring_instances_isolated_transports() -> None:
    transports: list[object] = []
    for index in range(10):
        wiring = resolve_application_runtime_event_delivery_wiring(
            _distributed_env(f"w5.f.iso.{index}"),
        )
        assert wiring.otlp_transport is not None
        transports.append(wiring.otlp_transport)
    assert len({id(t) for t in transports}) == 10


@pytest.mark.asyncio
async def test_collector_failure_isolated_from_runtime() -> None:
    class FailingCollector(OtlpTransportPort):
        def export(self, event: RuntimeEvent) -> None:
            raise DistributedTransportError("collector unavailable")

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    metrics = InternalDeliveryMetrics(exporter_kind="distributed_otlp")
    sink = OtlpEventExportSink(transport=FailingCollector())
    bridge = RuntimeEventExportSink(sink, delivery_metrics=metrics)
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
    assert metrics.snapshot().export_failed_total >= 1
    bus.close()


def test_collector_transport_flush_before_close() -> None:
    config = DistributedTransportConfiguration(
        endpoint=_COLLECTOR_ENDPOINT,
        protocol=OtlpProtocol.HTTP_PROTOBUF,
        service_name=_SERVICE_NAME,
        timeout_seconds=1.0,
    )
    transport = CollectorTransport(config)
    force_flush = MagicMock()
    shutdown = MagicMock()
    transport._inner._provider.force_flush = force_flush
    transport._inner._provider.shutdown = shutdown
    transport.close()
    force_flush.assert_called()
    shutdown.assert_called()


@pytest.mark.asyncio
async def test_backpressure_remains_bounded_with_slow_export() -> None:
    class SlowTransport(OtlpTransportPort):
        def export(self, event: RuntimeEvent) -> None:
            time.sleep(0.05)

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    bridge = RuntimeEventExportSink(OtlpEventExportSink(transport=SlowTransport()))
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=4))
    assert bounded._queue.maxsize == 4  # noqa: SLF001
    bus = RuntimeEventBus(event_sink=bounded)
    for index in range(20):
        await bus.publish(_event(f".bp.{index}"))
    await asyncio.sleep(0.3)
    bus.close()


@pytest.mark.asyncio
async def test_harness_runtime_wires_distributed_transport_without_execution_coupling() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = _distributed_env("w5.f.harness")
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    delivery = runtime.env_wiring.event_delivery
    assert delivery.otlp_transport is not None
    assert isinstance(delivery.otlp_transport, CollectorTransport)
    bus = runtime.env_wiring.build_context.runtime_event_bus
    await bus.publish(_event(".harness"))
    await asyncio.sleep(0.05)
    runtime.close()


@pytest.mark.parametrize(
    ("endpoint", "service_name", "timeout"),
    [
        ("", _SERVICE_NAME, 1.0),
        (_COLLECTOR_ENDPOINT, "", 1.0),
        (_COLLECTOR_ENDPOINT, _SERVICE_NAME, 0.0),
    ],
)
def test_distributed_configuration_validation(
    endpoint: str,
    service_name: str,
    timeout: float,
) -> None:
    config = DistributedTransportConfiguration(
        endpoint=endpoint,
        protocol=OtlpProtocol.HTTP_PROTOBUF,
        service_name=service_name,
        timeout_seconds=timeout,
    )
    with pytest.raises(ConfigurationError):
        validate_distributed_transport_configuration(config)


def test_factory_distributed_otlp_uses_otlp_event_sink() -> None:
    profile = ObservabilityExportProfile(
        enabled=True,
        exporter_kind=ExporterKind.DISTRIBUTED_OTLP,
    )
    sink = ObservabilityExportSinkFactory().create(profile)
    assert isinstance(sink, OtlpEventExportSink)
