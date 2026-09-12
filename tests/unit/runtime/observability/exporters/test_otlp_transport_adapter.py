# © Artur Czarnecki. All rights reserved.

"""W5-E — OTLP transport adapter qualification."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry.exporter.otlp.proto.http._log_exporter")

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
    OtlpExportConfiguration,
    OtlpProtocol,
    OtlpTransportError,
    OtlpTransportPort,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    NoopEventExportSink,
    OtlpEventExportSink,
    ObservabilityExportSinkFactory,
    RuntimeEventExportSink,
)
from intergrax.runtime.observability.exporters.otlp.otlp_configuration import (
    validate_otlp_export_configuration,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import (
    OtlpTransport,
    runtime_event_to_otlp_log_record,
)
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind="qualification.w5e",
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
            ObservabilityExportProfile(enabled=True, exporter_kind=ExporterKind.OTLP),
            OtlpEventExportSink,
        ),
    ],
)
def test_factory_selection(profile: ObservabilityExportProfile, expected_type: type) -> None:
    sink = ObservabilityExportSinkFactory().create(profile)
    assert isinstance(sink, expected_type)


def test_runtime_event_maps_to_otlp_log_record() -> None:
    event = _event()
    record = runtime_event_to_otlp_log_record(event)
    assert record.body
    attrs = record.attributes or {}
    assert attrs.get("intergrax.event_type") == event.event_type.value
    assert attrs.get("intergrax.run_id") == event.run_id


def test_flush_before_close_on_transport() -> None:
    config = OtlpExportConfiguration(
        endpoint="http://127.0.0.1:4318/v1/logs",
        protocol=OtlpProtocol.HTTP_PROTOBUF,
        timeout_seconds=1.0,
    )
    transport = OtlpTransport(config)
    force_flush = MagicMock()
    shutdown = MagicMock()
    transport._provider.force_flush = force_flush
    transport._provider.shutdown = shutdown
    transport.close()
    force_flush.assert_called()
    shutdown.assert_called()


@pytest.mark.asyncio
async def test_transport_failure_isolated_from_runtime_export_bridge() -> None:
    class FailingTransport(OtlpTransportPort):
        def export(self, event: RuntimeEvent) -> None:
            raise OtlpTransportError("simulated otlp outage")

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    metrics = InternalDeliveryMetrics(exporter_kind="otlp")
    sink = OtlpEventExportSink(transport=FailingTransport())
    bridge = RuntimeEventExportSink(sink, delivery_metrics=metrics)
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=16))
    bus = RuntimeEventBus(event_sink=bounded, delivery_metrics=metrics)
    handler_calls = 0

    def _handler(_event: RuntimeEvent) -> None:
        nonlocal handler_calls
        handler_calls += 1

    bus.subscribe(_handler, event_types={RuntimeEventType.TASK_PROGRESS})
    await bus.publish(_event())
    await asyncio.sleep(0.05)
    assert handler_calls == 1
    snap = metrics.snapshot()
    assert snap.export_failed_total >= 1
    bus.close()


@pytest.mark.parametrize(
    ("endpoint", "timeout"),
    [
        ("", 1.0),
        ("http://collector", 0.0),
        ("http://collector", -1.0),
    ],
)
def test_configuration_validation_rejects_invalid(endpoint: str, timeout: float) -> None:
    config = OtlpExportConfiguration(
        endpoint=endpoint,
        protocol=OtlpProtocol.HTTP_PROTOBUF,
        timeout_seconds=timeout,
    )
    with pytest.raises(ConfigurationError):
        validate_otlp_export_configuration(config)


def test_wiring_injects_otlp_transport_when_endpoint_configured() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w5.e.otlp")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.OTLP,
            "otlp_export_endpoint": "http://127.0.0.1:4318/v1/logs",
        },
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is not None
    assert isinstance(wiring.event_export_sink, OtlpEventExportSink)
