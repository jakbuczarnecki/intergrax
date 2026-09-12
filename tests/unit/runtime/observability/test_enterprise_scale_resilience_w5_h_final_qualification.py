# © Artur Czarnecki. All rights reserved.

"""W5-H — enterprise observability final qualification & deployment readiness."""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("opentelemetry.exporter.otlp.proto.http._log_exporter")

from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import GovernanceBundle
from intergrax.contracts.event_delivery import EventDeliveryPolicy, EventExportSinkPort
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.observability_export import (
    ExporterKind,
    OtlpTransportError,
    OtlpTransportPort,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    OtlpEventExportSink,
    RuntimeEventExportSink,
)
from intergrax.runtime.observability.exporters.distributed.collector_transport import (
    CollectorTransport,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTOR_ENDPOINT = "http://127.0.0.1:4318/v1/logs"
_SERVICE_NAME = "intergrax-runtime-w5h"


def _enterprise_env(profile_id: str) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    observability = GovernanceBundle.enterprise_cluster_observability(
        otlp_export_endpoint=_COLLECTOR_ENDPOINT,
        observability_export_service_name=_SERVICE_NAME,
    )
    return env.model_copy(
        update={
            "governance": env.governance.model_copy(update={"observability": observability}),
        },
    )


def _event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind="qualification.w5h",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def test_enterprise_profile_wires_collector_transport() -> None:
    wiring = resolve_application_runtime_event_delivery_wiring(_enterprise_env("w5.h.enterprise"))
    assert wiring.export_profile is not None
    assert wiring.export_profile.exporter_kind is ExporterKind.DISTRIBUTED_OTLP
    assert isinstance(wiring.otlp_transport, CollectorTransport)
    transport = wiring.otlp_transport
    assert transport._config.service_name == _SERVICE_NAME  # noqa: SLF001


def test_production_slo_regression_not_distributed_otlp() -> None:
    slo_obs = GovernanceBundle.production_slo().observability
    assert slo_obs.observability_exporter_kind is ExporterKind.OTLP
    assert slo_obs.observability_exporter_kind is not ExporterKind.DISTRIBUTED_OTLP
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w5.h.slo")
    env = env.model_copy(
        update={
            "governance": env.governance.model_copy(update={"observability": slo_obs}),
        },
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.export_profile is not None
    assert wiring.export_profile.exporter_kind is ExporterKind.OTLP
    if wiring.otlp_transport is not None:
        assert not isinstance(wiring.otlp_transport, CollectorTransport)


def test_runtime_instance_isolation_transports_and_sinks() -> None:
    wiring_a = resolve_application_runtime_event_delivery_wiring(_enterprise_env("w5.h.iso.a"))
    wiring_b = resolve_application_runtime_event_delivery_wiring(_enterprise_env("w5.h.iso.b"))
    assert wiring_a.otlp_transport is not wiring_b.otlp_transport
    assert wiring_a.event_export_sink is not wiring_b.event_export_sink
    assert wiring_a.bounded_sink is not wiring_b.bounded_sink


@pytest.mark.asyncio
async def test_failure_containment_otlp_transport_error() -> None:
    class FailingCollector(OtlpTransportPort):
        def export(self, event: RuntimeEvent) -> None:
            raise OtlpTransportError("collector unavailable")

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    metrics = InternalDeliveryMetrics(exporter_kind=ExporterKind.DISTRIBUTED_OTLP.value)
    bridge = RuntimeEventExportSink(
        OtlpEventExportSink(transport=FailingCollector()),
        delivery_metrics=metrics,
    )
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
    assert metrics.snapshot().export_failed_total >= 1
    bus.close()


@pytest.mark.asyncio
async def test_backpressure_qualification_slow_exporter() -> None:
    class SlowTransport(OtlpTransportPort):
        def __init__(self) -> None:
            self._export_calls = 0

        def export(self, event: RuntimeEvent) -> None:
            self._export_calls += 1
            if self._export_calls == 1:
                time.sleep(5.0)

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    max_capacity = 4
    bridge = RuntimeEventExportSink(OtlpEventExportSink(transport=SlowTransport()))
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=max_capacity))
    assert bounded._queue.maxsize == max_capacity  # noqa: SLF001
    bus = RuntimeEventBus(event_sink=bounded)
    for index in range(24):
        await bus.publish(_event())
        if bounded._queue.qsize() > max_capacity:  # noqa: SLF001
            pytest.fail("bounded queue exceeded configured max_capacity")
    await asyncio.sleep(5.5)
    bus.close()


class _LifecycleProbeExportSink(EventExportSinkPort):
    def __init__(self) -> None:
        self.events: list[str] = []

    async def export(self, event: RuntimeEvent) -> None:
        return

    async def flush(self) -> None:
        self.events.append("flush")

    async def close(self) -> None:
        self.events.append("close")


def test_lifecycle_ordering_flush_before_close() -> None:
    probe = _LifecycleProbeExportSink()
    bridge = RuntimeEventExportSink(probe)
    bounded = BoundedEventSink(bridge, EventDeliveryPolicy(max_capacity=8))
    bus = RuntimeEventBus(event_sink=bounded)
    bus.close()
    assert probe.events == ["flush", "close"]
