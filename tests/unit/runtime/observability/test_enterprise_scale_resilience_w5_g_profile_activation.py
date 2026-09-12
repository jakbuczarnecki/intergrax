# © Artur Czarnecki. All rights reserved.

"""W5-G — enterprise distributed observability profile activation & qualification."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import GovernanceBundle
from intergrax.contracts.event_delivery import EventDeliveryPolicy
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.observability_export import (
    ConfigurationError,
    ExporterKind,
    OtlpProtocol,
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
from intergrax.runtime.observability.exporters.distributed.distributed_configuration import (
    DistributedTransportConfiguration,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_COLLECTOR_ENDPOINT = "http://127.0.0.1:4318/v1/logs"
_SERVICE_NAME = "intergrax-runtime"


def _enterprise_observability_env(profile_id: str) -> ApplicationEnvironmentProfile:
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


def _local_lab_env(profile_id: str) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)


def _event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_PROGRESS,
        phase=ExecutionPhase.COMPLETION,
        event_kind="qualification.w5g",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def test_enterprise_profile_selects_distributed_otlp() -> None:
    env = _enterprise_observability_env("w5.g.enterprise")
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.export_profile is not None
    assert wiring.export_profile.exporter_kind is ExporterKind.DISTRIBUTED_OTLP
    assert isinstance(wiring.otlp_transport, CollectorTransport)


def test_local_profile_stays_noop_or_recording_without_bounded_delivery() -> None:
    env = _local_lab_env("w5.g.local")
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.bounded_sink is None
    assert wiring.export_profile is None
    assert wiring.otlp_transport is None


def test_invalid_distributed_configuration_raises_configuration_error() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w5.g.invalid")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.DISTRIBUTED_OTLP,
            "otlp_export_endpoint": "",
            "observability_export_service_name": _SERVICE_NAME,
        },
    )
    with pytest.raises(ConfigurationError):
        resolve_application_runtime_event_delivery_wiring(env)


def test_environment_instance_isolation() -> None:
    wiring_a = resolve_application_runtime_event_delivery_wiring(
        _enterprise_observability_env("w5.g.iso.a"),
    )
    wiring_b = resolve_application_runtime_event_delivery_wiring(
        _enterprise_observability_env("w5.g.iso.b"),
    )
    assert wiring_a.bounded_sink is not None
    assert wiring_b.bounded_sink is not None
    assert wiring_a.bounded_sink is not wiring_b.bounded_sink
    assert wiring_a.otlp_transport is not wiring_b.otlp_transport


def test_shutdown_flush_before_transport_close() -> None:
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
async def test_export_failure_isolated_from_execution_plane() -> None:
    class FailingTransport(OtlpTransportPort):
        def export(self, event: RuntimeEvent) -> None:
            raise OtlpTransportError("collector unavailable")

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    metrics = InternalDeliveryMetrics(exporter_kind=ExporterKind.DISTRIBUTED_OTLP.value)
    bridge = RuntimeEventExportSink(
        OtlpEventExportSink(transport=FailingTransport()),
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
