# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunStats,
    SerializedTraceEvent,
)

pytestmark = pytest.mark.gate


def _persisted_run(*events: SerializedTraceEvent) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id="run-journal-1",
            session_id="s1",
            user_id="u1",
            tenant_id="tenant-a",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=list(events),
    )


def test_unified_journal_trace_only_when_store_missing() -> None:
    trace = SerializedTraceEvent(
        event_id="trace-1",
        run_id="run-journal-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> completed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": "run-journal-1", "task_state": "completed"},
        artifact_refs=[],
    )
    journal = build_unified_run_journal(_persisted_run(trace), runtime_store=None)
    assert len(journal) == 1
    assert journal[0].event_type == RuntimeEventType.TASK_COMPLETED
    assert journal[0].event_id == "rt_trace-1"


def test_unified_journal_merges_persisted_and_trace_without_duplicates() -> None:
    trace = SerializedTraceEvent(
        event_id="trace-2",
        run_id="run-journal-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="tools",
        step="tool_invocation_start",
        message="invoke jira.search_tasks",
        payload_schema_id=None,
        payload_schema_version=None,
        payload={"tool_name": "jira.search_tasks"},
        tags={"task_id": "run-journal-1", "tool_name": "jira.search_tasks"},
        artifact_refs=[],
    )
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id="evt_policy_1",
            tenant_id="tenant-a",
            task_id="run-journal-1",
            run_id="run-journal-1",
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            severity=EventSeverity.INFO,
            payload={"policy_action": "allow", "source": "uaep"},
            timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
            correlation_id="run-journal-1",
        ),
        tenant_id="tenant-a",
    )
    store.append(
        RuntimeEvent(
            event_id="rt_trace-2",
            tenant_id="tenant-a",
            task_id="run-journal-1",
            run_id="run-journal-1",
            event_type=RuntimeEventType.TOOL_REQUESTED,
            phase=ExecutionPhase.STEP_EXECUTION,
            severity=EventSeverity.INFO,
            payload={"trace_event_id": "trace-2", "tool_name": "jira.search_tasks"},
            timestamp=datetime(2026, 6, 7, 10, 0, 1, tzinfo=timezone.utc),
            correlation_id="run-journal-1",
        ),
        tenant_id="tenant-a",
    )

    journal = build_unified_run_journal(_persisted_run(trace), runtime_store=store)
    event_ids = [event.event_id for event in journal]
    assert event_ids == ["evt_policy_1", "rt_trace-2"]
    assert journal[0].event_type == RuntimeEventType.POLICY_DECISION
    assert journal[1].event_type == RuntimeEventType.TOOL_REQUESTED


def test_unified_journal_sorts_chronologically() -> None:
    trace_late = SerializedTraceEvent(
        event_id="trace-late",
        run_id="run-journal-1",
        seq=2,
        ts_utc="2026-06-07T10:00:03+00:00",
        level="info",
        component="engine",
        step="core_llm",
        message="adapter returned",
        payload_schema_id=CoreLLMCallRecordedDiagV1.schema_id(),
        payload_schema_version=1,
        payload={
            "model": "gpt-test",
            "provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "finish_reason": "stop",
            "response_id": "resp-1",
            "has_refusal": False,
            "has_tool_calls": False,
        },
        tags={"task_id": "run-journal-1"},
        artifact_refs=[],
    )
    trace_early = SerializedTraceEvent(
        event_id="trace-early",
        run_id="run-journal-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> running",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": "run-journal-1", "task_state": "running"},
        artifact_refs=[],
    )
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id="evt_mid",
            tenant_id="tenant-a",
            task_id="run-journal-1",
            run_id="run-journal-1",
            event_type=RuntimeEventType.LLM_CALL,
            phase=ExecutionPhase.STEP_EXECUTION,
            severity=EventSeverity.INFO,
            payload={"model": "gpt-live", "source": "bus"},
            timestamp=datetime(2026, 6, 7, 10, 0, 2, tzinfo=timezone.utc),
            correlation_id="run-journal-1",
        ),
        tenant_id="tenant-a",
    )

    journal = build_unified_run_journal(
        _persisted_run(trace_early, trace_late),
        runtime_store=store,
    )
    assert [event.event_type for event in journal] == [
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.LLM_CALL,
        RuntimeEventType.LLM_CALL,
    ]
    assert journal[0].event_id == "rt_trace-early"
    assert journal[1].event_id == "evt_mid"
    assert journal[2].event_id == "rt_trace-late"
