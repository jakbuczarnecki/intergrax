# © Artur Czarnecki. All rights reserved.

"""HARDEN-3C — exporter failure, latency, recovery, and bus integration semantics."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import mint_event_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.hosting.contracts.events import HostedApplicationEvent, HostedApplicationEventType
from intergrax.hosting import HostedApplicationLifecycleState
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_boundary import (
    ExportRecordKind,
    InMemoryObservabilityExporter,
    ObservabilityExportEnvelope,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin
from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporterConfig
from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = pytest.mark.unit


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
        self.exported_event_ids: list[str] = []

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise RuntimeError("temporary outage")
        self.exported_event_ids.append(envelope.event_id)


class _SlowObservabilityExporter:
    def __init__(self, *, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.started.set()
        await self.release.wait()
        await asyncio.sleep(self.delay_seconds)


class _OrderingProbeExporter:
    def __init__(self, runtime_store: InMemoryRuntimeEventStore, *, run_id: str) -> None:
        self.runtime_store = runtime_store
        self.run_id = run_id
        self.persisted_count_at_export: int | None = None

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        self.persisted_count_at_export = len(
            self.runtime_store.list_for_run(self.run_id, tenant_id="tenant-a")
        )
        raise RuntimeError("export sink unavailable")


class _BlockingObservabilityExporter:
    def __init__(self, *, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    async def export(self, envelope: ObservabilityExportEnvelope) -> None:
        await asyncio.sleep(self.delay_seconds)


def _runtime_event(
    *,
    event_id: str | None = None,
    identity: dict[str, object] | None = None,
) -> RuntimeEvent:
    resolved_identity = identity or runtime_event_test_identity()
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id="tenant-a",
        event_type=RuntimeEventType.TOOL_COMPLETED,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={"tool_id": "workspace.read_file", "latency_ms": 3},
        **resolved_identity,
    )


@pytest.mark.asyncio
async def test_canonical_persistence_occurs_before_export_attempt() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    probe = _OrderingProbeExporter(runtime_store, run_id=identity["run_id"])
    plugin = make_observability_export_runtime_plugin(
        exporter=probe,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    event = _runtime_event(identity=identity)
    await bus.publish(event)

    assert probe.persisted_count_at_export == 1
    persisted = runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")
    assert len(persisted) == 1
    assert persisted[0].event_id == event.event_id


@pytest.mark.asyncio
async def test_exporter_failure_does_not_remove_canonical_event() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    exporter = _FailingObservabilityExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    await bus.publish(_runtime_event(identity=identity))

    persisted = runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")
    assert len(persisted) == 1
    assert exporter.call_count == 1


@pytest.mark.asyncio
async def test_repeated_exporter_outage_persists_all_events_and_classifies_failures() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    exporter = _FailingObservabilityExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    for index in range(3):
        await bus.publish(_runtime_event(event_id=mint_event_id(), identity=identity))

    persisted = runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")
    assert len(persisted) == 3
    assert exporter.call_count == 3


@pytest.mark.asyncio
async def test_exporter_recovery_exports_new_events_without_replay() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    exporter = _RecoveringObservabilityExporter(fail_count=2)
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    event_id_1 = mint_event_id()
    event_id_2 = mint_event_id()
    event_id_3 = mint_event_id()
    await bus.publish(_runtime_event(event_id=event_id_1, identity=identity))
    await bus.publish(_runtime_event(event_id=event_id_2, identity=identity))
    await bus.publish(_runtime_event(event_id=event_id_3, identity=identity))

    assert exporter.call_count == 3
    assert exporter.exported_event_ids == [event_id_3]


@pytest.mark.asyncio
async def test_exporter_failure_does_not_create_recursive_export_events() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=runtime_store, record_history=False)
    identity = runtime_event_test_identity()
    exporter = _FailingObservabilityExporter()
    plugin = make_observability_export_runtime_plugin(
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    await bus.publish(_runtime_event(identity=identity))

    assert exporter.call_count == 1
    assert len(runtime_store.list_for_run(identity["run_id"], tenant_id="tenant-a")) == 1


@pytest.mark.asyncio
async def test_publish_awaits_slow_exporter_inline() -> None:
    bus = RuntimeEventBus(record_history=False)
    slow = _SlowObservabilityExporter(delay_seconds=0.05)
    plugin = make_observability_export_runtime_plugin(
        exporter=slow,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    publish_task = asyncio.create_task(bus.publish(_runtime_event()))
    await slow.started.wait()
    assert publish_task.done() is False

    slow.release.set()
    await publish_task
    assert publish_task.done() is True


def test_record_blocks_sync_caller_until_slow_exporter_finishes() -> None:
    bus = RuntimeEventBus(record_history=False)
    blocking = _BlockingObservabilityExporter(delay_seconds=0.1)
    plugin = make_observability_export_runtime_plugin(
        exporter=blocking,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    started = time.perf_counter()
    bus.record(_runtime_event())
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.09


@pytest.mark.asyncio
async def test_otlp_transport_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock()
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)
    client.aclose = AsyncMock()

    def _fake_async_client(*, timeout: float) -> httpx.AsyncClient:
        captured["timeout"] = timeout
        return client

    monkeypatch.setattr(httpx, "AsyncClient", _fake_async_client)
    transport = OtlpHttpTransport()
    config = OtlpObservabilityExporterConfig(
        endpoint="https://collector.example/v1/logs",
        service_name="intergrax.test",
        timeout_seconds=2.5,
    )

    await transport.send({"resourceLogs": []}, config=config)

    assert captured["timeout"] == 2.5


@pytest.mark.asyncio
async def test_try_export_classifies_exporter_exception_as_failed() -> None:
    exporter = _FailingObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )

    assert result.exported is False
    assert result.reason == "exporter_failed"


@pytest.mark.asyncio
async def test_platform_signal_export_failure_is_isolated() -> None:
    exporter = _FailingObservabilityExporter()
    publisher = ObservabilityHostedApplicationEventPublisher(
        exporter,
        policy=ObservabilityExportPolicy(enabled=True),
    )
    event = HostedApplicationEvent(
        event_id="hos-evt-1",
        event_type=HostedApplicationEventType.APPLICATION_READY,
        occurred_at=datetime(2026, 8, 30, 8, 0, tzinfo=UTC),
        application_id="test_app",
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
        correlation_id="corr-1",
        severity=EventSeverity.INFO,
        payload={"component_id": "worker-1"},
    )

    await publisher.publish(event)

    assert exporter.call_count == 1


@pytest.mark.asyncio
async def test_disabled_export_policy_skips_exporter_without_false_success() -> None:
    exporter = InMemoryObservabilityExporter()
    envelope = ObservabilityExportEnvelope(record_kind=ExportRecordKind.RUNTIME_EVENT, run_id="run-1")

    result = await try_export_observability_envelope(
        envelope,
        exporter=exporter,
        policy=ObservabilityExportPolicy(enabled=False),
    )

    assert result.exported is False
    assert result.reason == "export_disabled"
    assert exporter.envelopes == []
