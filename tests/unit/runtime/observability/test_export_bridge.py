# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
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
    run_id = "run-bridge-1"
    tenant_id = "tenant-a"
    _seed_trace(store, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus(record_history=False)
    plugin = make_journal_export_runtime_plugin(trace_store=store)
    plugin.register(bus, HookRegistry(), MagicMock())

    event = RuntimeEvent(
        task_id=run_id,
        run_id=run_id,
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
