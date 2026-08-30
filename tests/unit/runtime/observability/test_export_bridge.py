# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import mint_attempt_id, mint_event_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats, SerializedTraceEvent
from intergrax.runtime.observability.export_bridge import make_journal_export_runtime_plugin

pytestmark = pytest.mark.gate


def _seed_trace(store: InMemoryRunTraceStore, *, run_id: str, tenant_id: str) -> None:
    store._events_by_run[run_id] = [
        SerializedTraceEvent(
            event_id="trace-1",
            run_id=run_id,
            seq=1,
            ts_utc="2026-06-07T10:00:01+00:00",
            level="info",
            component="planner",
            step="task_lifecycle",
            message="task state -> completed",
            payload_schema_id=None,
            payload_schema_version=None,
            payload=None,
            tags={"task_id": run_id, "task_state": "completed"},
            artifact_refs=[],
        ),
    ]
    store.finalize_run(
        run_id,
        RunMetadata(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id="user-1",
            session_id="session-1",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=5, llm_usage={}),
        ),
    )


@pytest.mark.asyncio
async def test_journal_export_plugin_handles_task_completed() -> None:
    store = InMemoryRunTraceStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    tenant_id = "tenant-a"
    _seed_trace(store, run_id=run_id, tenant_id=tenant_id)

    runtime_store = InMemoryRuntimeEventStore()
    runtime_store.append(
        RuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.TASK_COMPLETED,
            phase=ExecutionPhase.COMPLETION,
            severity=EventSeverity.INFO,
            payload={},
            timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
            correlation_id=run_id,
        ),
        tenant_id=tenant_id,
    )

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(
        trace_store=store,
        runtime_event_store=runtime_store,
    )
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        event_id=mint_event_id(),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
        agent_id="agent-1",
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={"journal_ref": {"event_count": 1}},
    )

    with patch("intergrax.runtime.observability.export_bridge.logger") as mock_logger:
        with patch(
            "intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"
        ) as mock_parser_export:
            await bus.publish(event)

    mock_parser_export.assert_called_once()
    mock_logger.info.assert_called_once()
    extra = mock_logger.info.call_args.kwargs["extra"]
    assert extra["journal_otlp"]["resourceSpans"]
    assert extra["journal_export"]["event_count"] == 1


@pytest.mark.asyncio
async def test_journal_export_queries_trace_reader_with_canonical_run_id() -> None:
    store = InMemoryRunTraceStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    assert task_id != run_id
    tenant_id = "tenant-a"
    _seed_trace(store, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=store)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        event_id=mint_event_id(),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={},
    )

    with patch.object(store, "read_run", wraps=store.read_run) as mock_read_run:
        with patch("intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"):
            await bus.publish(event)

    mock_read_run.assert_called_once_with(run_id, tenant_id)


@pytest.mark.asyncio
async def test_journal_export_uses_exact_tenant_id() -> None:
    store = InMemoryRunTraceStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    tenant_id = "tenant-a"
    _seed_trace(store, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=store)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        event_id=mint_event_id(),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={},
    )

    with patch.object(store, "read_run", wraps=store.read_run) as mock_read_run:
        with patch("intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"):
            await bus.publish(event)

    _args, kwargs = mock_read_run.call_args
    assert _args == (run_id, "tenant-a")
    assert kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", [None, "", "   "])
async def test_journal_export_skips_missing_tenant_without_default(tenant_id: str | None) -> None:
    store = InMemoryRunTraceStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _seed_trace(store, run_id=run_id, tenant_id="tenant-a")

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=store)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        event_id=mint_event_id(),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={},
    )

    with patch.object(store, "read_run", wraps=store.read_run) as mock_read_run:
        with patch(
            "intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"
        ) as mock_parser_export:
            await bus.publish(event)

    mock_read_run.assert_not_called()
    mock_parser_export.assert_not_called()


@pytest.mark.asyncio
async def test_journal_export_parser_traces_without_runtime_event_store() -> None:
    store = InMemoryRunTraceStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    tenant_id = "tenant-a"
    _seed_trace(store, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=store, runtime_event_store=None)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        event_id=mint_event_id(),
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        payload={},
    )

    with patch("intergrax.runtime.observability.export_bridge.logger") as mock_logger:
        with patch(
            "intergrax.runtime.observability.export_bridge.export_parser_traces_from_events"
        ) as mock_parser_export:
            await bus.publish(event)

    mock_parser_export.assert_called_once()
    mock_logger.info.assert_not_called()


def test_journal_export_source_has_no_identity_fallbacks() -> None:
    source = Path("intergrax/runtime/observability/export_bridge.py").read_text(encoding="utf-8")
    forbidden = [
        "event.run_id or event.task_id",
        "event.task_id or event.run_id",
        'event.tenant_id or "default"',
        'tenant_id or "default"',
    ]
    for pattern in forbidden:
        assert pattern not in source
