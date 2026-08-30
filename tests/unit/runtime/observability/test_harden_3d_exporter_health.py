# © Artur Czarnecki. All rights reserved.

"""HARDEN-3D — exporter health state machine and export path integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_event_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_health import (
    ObservabilityExporterHealthRegistry,
    ObservabilityExporterHealthStatus,
    normalize_export_failure_reason,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.export_routing import (
    FanoutObservabilityExporter,
    ObservabilityExportRoute,
)
from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(seconds=1)


def test_h1_no_attempt_has_no_snapshot() -> None:
    registry = ObservabilityExporterHealthRegistry()

    assert registry.get("otlp-primary") is None
    assert registry.list() == ()


def test_h2_success_records_healthy_snapshot() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_success("otlp-primary", _T0)
    snapshot = registry.get("otlp-primary")

    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.HEALTHY
    assert snapshot.consecutive_failures == 0
    assert snapshot.last_attempt_at == _T0
    assert snapshot.last_success_at == _T0
    assert snapshot.last_failure_at is None
    assert snapshot.last_failure_reason is None
    assert snapshot.recovery_count == 0


def test_h3_first_failure_records_degraded_snapshot() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_failure("otlp-primary", "exporter_failed", _T0)
    snapshot = registry.get("otlp-primary")

    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.DEGRADED
    assert snapshot.consecutive_failures == 1
    assert snapshot.last_failure_at == _T0
    assert snapshot.last_failure_reason == "exporter_failed"
    assert snapshot.last_success_at is None


def test_h4_repeated_failure_increments_consecutive_failures() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_failure("otlp-primary", "exporter_failed", _T0)
    registry.record_failure("otlp-primary", "timeout", _T1)
    snapshot = registry.get("otlp-primary")

    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.DEGRADED
    assert snapshot.consecutive_failures == 2
    assert snapshot.last_failure_at == _T1
    assert snapshot.last_failure_reason == "timeout"


def test_h5_failure_then_success_recovers_to_healthy() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_failure("otlp-primary", "exporter_failed", _T0)
    registry.record_success("otlp-primary", _T1)
    snapshot = registry.get("otlp-primary")

    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.HEALTHY
    assert snapshot.consecutive_failures == 0
    assert snapshot.last_success_at == _T1
    assert snapshot.last_failure_at == _T0
    assert snapshot.last_failure_reason == "exporter_failed"
    assert snapshot.recovery_count == 1


def test_h6_healthy_then_healthy_updates_success_timestamps() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_success("otlp-primary", _T0)
    registry.record_success("otlp-primary", _T1)
    snapshot = registry.get("otlp-primary")

    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.HEALTHY
    assert snapshot.consecutive_failures == 0
    assert snapshot.last_attempt_at == _T1
    assert snapshot.last_success_at == _T1
    assert snapshot.recovery_count == 0


def test_normalize_export_failure_reason_rejects_vendor_specific_details() -> None:
    assert normalize_export_failure_reason("timeout") == "timeout"
    assert normalize_export_failure_reason("transport_error") == "transport_error"
    assert normalize_export_failure_reason("Connection refused to collector.example:4317") == "exporter_failed"


def test_list_returns_all_exporter_snapshots() -> None:
    registry = ObservabilityExporterHealthRegistry()

    registry.record_success("route-a", _T0)
    registry.record_failure("route-b", "exporter_failed", _T1)

    snapshots = registry.list()
    assert {item.exporter_id for item in snapshots} == {"route-a", "route-b"}


class _FailingObservabilityExporter:
    def __init__(self) -> None:
        self.call_count = 0

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.call_count += 1
        raise RuntimeError("export sink unavailable")


class _RecoveringObservabilityExporter:
    def __init__(self, *, fail_count: int) -> None:
        self._fail_count = fail_count
        self.call_count = 0

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError("temporary outage")


def _runtime_event(*, identity: dict[str, object] | None = None) -> RuntimeEvent:
    resolved_identity = identity or runtime_event_test_identity()
    return RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id="tenant-a",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={"tool_id": "workspace.read_file", "latency_ms": 3},
        **resolved_identity,
    )


@pytest.mark.asyncio
async def test_h7_disabled_policy_does_not_mutate_health_registry() -> None:
    registry = ObservabilityExporterHealthRegistry()
    exporter = _FailingObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=False),
        health_registry=registry,
        exporter_id="otlp-primary",
    )

    assert registry.get("otlp-primary") is None
    assert exporter.call_count == 0


@pytest.mark.asyncio
async def test_h7_filtered_policy_does_not_mutate_health_registry() -> None:
    registry = ObservabilityExporterHealthRegistry()
    exporter = _FailingObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True, export_content=True),
        health_registry=registry,
        exporter_id="otlp-primary",
    )

    assert registry.get("otlp-primary") is None
    assert exporter.call_count == 0


@pytest.mark.asyncio
async def test_bus_integration_marks_exporter_degraded_without_affecting_persistence() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    registry = ObservabilityExporterHealthRegistry()
    exporter = _FailingObservabilityExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
        health_registry=registry,
        exporter_id="hos-primary",
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    await bus.publish(_runtime_event(identity=identity))

    persisted = runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")
    assert len(persisted) == 1
    assert exporter.call_count == 1
    snapshot = registry.get("hos-primary")
    assert snapshot is not None
    assert snapshot.status is ObservabilityExporterHealthStatus.DEGRADED
    assert snapshot.consecutive_failures == 1


@pytest.mark.asyncio
async def test_bus_integration_recovers_exporter_health_after_success() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    registry = ObservabilityExporterHealthRegistry()
    exporter = _RecoveringObservabilityExporter(fail_count=1)
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
        health_registry=registry,
        exporter_id="hos-primary",
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    await bus.publish(_runtime_event(identity=identity))
    degraded = registry.get("hos-primary")
    assert degraded is not None
    assert degraded.status is ObservabilityExporterHealthStatus.DEGRADED

    await bus.publish(_runtime_event(identity=identity))
    recovered = registry.get("hos-primary")
    assert recovered is not None
    assert recovered.status is ObservabilityExporterHealthStatus.HEALTHY
    assert recovered.consecutive_failures == 0
    assert recovered.last_failure_at is not None
    assert recovered.last_success_at is not None
    assert recovered.last_failure_at <= recovered.last_success_at
    assert len(runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")) == 2


@pytest.mark.asyncio
async def test_health_update_does_not_emit_additional_runtime_events() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    registry = ObservabilityExporterHealthRegistry()
    exporter = _FailingObservabilityExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
        health_registry=registry,
        exporter_id="hos-primary",
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    await bus.publish(_runtime_event(identity=identity))

    assert exporter.call_count == 1
    assert len(runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")) == 1


@pytest.mark.asyncio
async def test_fanout_records_per_route_health_without_global_corruption() -> None:
    registry = ObservabilityExporterHealthRegistry()
    route_a = InMemoryObservabilityExporter()
    route_b = _FailingObservabilityExporter()
    route_c = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [
            ObservabilityExportRoute(route_id="route-a", exporter=route_a),
            ObservabilityExportRoute(route_id="route-b", exporter=route_b),
            ObservabilityExportRoute(route_id="route-c", exporter=route_c),
        ],
        health_registry=registry,
    )
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    await try_export_observability_envelope(
        envelope,
        exporter=fanout,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    snapshot_a = registry.get("route-a")
    snapshot_b = registry.get("route-b")
    snapshot_c = registry.get("route-c")
    assert snapshot_a is not None
    assert snapshot_a.status is ObservabilityExporterHealthStatus.HEALTHY
    assert snapshot_b is not None
    assert snapshot_b.status is ObservabilityExporterHealthStatus.DEGRADED
    assert snapshot_c is not None
    assert snapshot_c.status is ObservabilityExporterHealthStatus.HEALTHY
    assert len(route_a.envelopes) == 1
    assert route_b.call_count == 1
    assert len(route_c.envelopes) == 1
    assert fanout.last_result is not None
    assert fanout.last_result.exported_count == 2
    assert fanout.last_result.failed_count == 1


@pytest.mark.asyncio
async def test_fanout_skipped_routes_do_not_create_health_snapshots() -> None:
    registry = ObservabilityExporterHealthRegistry()
    inner = InMemoryObservabilityExporter()
    fanout = FanoutObservabilityExporter(
        [
            ObservabilityExportRoute(
                route_id="problem-only",
                exporter=inner,
                enabled=False,
            )
        ],
        health_registry=registry,
    )
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    await fanout.export(envelope)

    assert registry.get("problem-only") is None
