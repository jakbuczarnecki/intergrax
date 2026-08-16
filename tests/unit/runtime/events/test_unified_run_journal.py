# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunStats,
    SerializedTraceEvent,
)

pytestmark = pytest.mark.gate

_TENANT = "tenant-a"
_JOURNAL_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "events"
    / "unified_run_journal.py"
)


class _RecordingStore(InMemoryRuntimeEventStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_for_run_calls: list[tuple[str, str, int]] = []

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> list[RuntimeEvent]:
        self.list_for_run_calls.append((run_id, tenant_id, limit))
        return super().list_for_run(run_id, tenant_id=tenant_id, limit=limit)


def _persisted_run(
    *events: SerializedTraceEvent,
    run_id: str,
    tenant_id: str = _TENANT,
) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=list(events),
    )


def _trace(*, run_id: str, event_id: str = "trace-row") -> SerializedTraceEvent:
    return SerializedTraceEvent(
        event_id=event_id,
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
    )


def _event(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    event_type: RuntimeEventType,
    event_id: str | None = None,
    timestamp: datetime | None = None,
    payload: dict[str, object] | None = None,
    tenant_id: str = _TENANT,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        payload=dict(payload or {}),
        timestamp=timestamp or datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        correlation_id=run_id,
    )


def test_unified_journal_preserves_canonical_identity() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_id = mint_event_id()
    store = InMemoryRuntimeEventStore()
    stored = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TASK_CREATED,
        event_id=event_id,
    )
    store.append(stored, tenant_id=_TENANT)

    journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
    assert len(journal) == 1
    assert journal[0].task_id == task_id
    assert journal[0].run_id == run_id
    assert journal[0].attempt_id == attempt_id
    assert journal[0].event_id == event_id
    assert journal[0] is stored


def test_unified_journal_does_not_convert_trace_rows_without_store() -> None:
    run_id = mint_run_id()
    journal = build_unified_run_journal(
        _persisted_run(_trace(run_id=run_id), run_id=run_id),
        runtime_store=None,
    )
    assert journal == []


def test_unified_journal_does_not_substitute_run_id_for_missing_task_id() -> None:
    run_id = mint_run_id()
    journal = build_unified_run_journal(
        _persisted_run(_trace(run_id=run_id), run_id=run_id),
        runtime_store=InMemoryRuntimeEventStore(),
    )
    assert journal == []


def test_unified_journal_rejects_task_id_as_run_id() -> None:
    store = InMemoryRuntimeEventStore()
    with pytest.raises(ValueError, match="RunId must start with"):
        build_unified_run_journal(_persisted_run(run_id=mint_task_id()), runtime_store=store)


def test_unified_journal_rejects_attempt_id_as_run_id() -> None:
    with pytest.raises(ValueError, match="RunId must start with"):
        build_unified_run_journal(_persisted_run(run_id=mint_attempt_id()), runtime_store=None)


def test_unified_journal_rejects_event_id_as_run_id() -> None:
    with pytest.raises(ValueError, match="RunId must start with"):
        build_unified_run_journal(_persisted_run(run_id=mint_event_id()), runtime_store=None)


def test_malformed_identity_prefixes_fail_validators() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    with pytest.raises(ValueError, match="TaskId must start with"):
        _event(
            task_id=run_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
        )
    with pytest.raises(ValueError, match="RunId must start with"):
        _event(
            task_id=task_id,
            run_id=task_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
        )
    with pytest.raises(ValueError, match="AttemptId must start with"):
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=run_id,
            event_type=RuntimeEventType.TASK_CREATED,
        )
    with pytest.raises(ValueError, match="EventId must start with"):
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
            event_id=task_id,
        )


def test_unified_journal_preserves_retry_segmentation() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    events = [
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
            event_type=RuntimeEventType.TASK_CREATED,
            timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        ),
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
            event_type=RuntimeEventType.RETRY_SCHEDULED,
            timestamp=datetime(2026, 6, 7, 10, 0, 1, tzinfo=timezone.utc),
        ),
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a2,
            event_type=RuntimeEventType.RETRY_STARTED,
            timestamp=datetime(2026, 6, 7, 10, 0, 2, tzinfo=timezone.utc),
        ),
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a2,
            event_type=RuntimeEventType.STEP_STARTED,
            timestamp=datetime(2026, 6, 7, 10, 0, 3, tzinfo=timezone.utc),
        ),
    ]
    for event in events:
        store.append(event, tenant_id=_TENANT)

    journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
    assert [event.event_type for event in journal] == [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.RETRY_SCHEDULED,
        RuntimeEventType.RETRY_STARTED,
        RuntimeEventType.STEP_STARTED,
    ]
    assert [event.attempt_id for event in journal] == [
        attempt_a1,
        attempt_a1,
        attempt_a2,
        attempt_a2,
    ]


def test_unified_journal_preserves_multi_retry_attempts() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempts = (mint_attempt_id(), mint_attempt_id(), mint_attempt_id())
    store = InMemoryRuntimeEventStore()
    for index, attempt_id in enumerate(attempts):
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.STEP_STARTED,
                timestamp=datetime(2026, 6, 7, 10, 0, index, tzinfo=timezone.utc),
            ),
            tenant_id=_TENANT,
        )

    journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
    assert [event.attempt_id for event in journal] == list(attempts)


def test_unified_journal_resume_preserves_same_attempt() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    sequence = (
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.PAUSED,
        RuntimeEventType.RESUMED,
        RuntimeEventType.STEP_COMPLETED,
    )
    for index, event_type in enumerate(sequence):
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=event_type,
                timestamp=datetime(2026, 6, 7, 10, 0, index, tzinfo=timezone.utc),
            ),
            tenant_id=_TENANT,
        )

    journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
    assert [event.event_type for event in journal] == list(sequence)
    assert {event.attempt_id for event in journal} == {attempt_id}


def test_unified_journal_does_not_duplicate_stored_events_from_traces() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_id = mint_event_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TOOL_REQUESTED,
            event_id=event_id,
            payload={"trace_event_id": "trace-2"},
        ),
        tenant_id=_TENANT,
    )
    journal = build_unified_run_journal(
        _persisted_run(_trace(run_id=run_id, event_id="trace-2"), run_id=run_id),
        runtime_store=store,
    )
    assert [event.event_id for event in journal] == [event_id]


def test_unified_journal_queries_exact_persisted_tenant() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    store = _RecordingStore()
    store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.POLICY_DECISION,
        ),
        tenant_id=_TENANT,
    )
    journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
    assert len(journal) == 1
    assert store.list_for_run_calls == [(run_id, _TENANT, 2000)]


def test_unified_journal_rejects_empty_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id is required"):
        build_unified_run_journal(
            _persisted_run(run_id=mint_run_id(), tenant_id=""),
            runtime_store=InMemoryRuntimeEventStore(),
        )


def test_unified_journal_rejects_non_positive_limit() -> None:
    persisted = _persisted_run(run_id=mint_run_id())
    store = InMemoryRuntimeEventStore()
    with pytest.raises(ValueError, match="journal limit must be > 0"):
        build_unified_run_journal(persisted, runtime_store=store, limit=0)
    with pytest.raises(ValueError, match="journal limit must be > 0"):
        build_unified_run_journal(persisted, runtime_store=store, limit=-1)


def test_unified_journal_does_not_use_active_execution_identity() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_a1,
            event_type=RuntimeEventType.STEP_STARTED,
        ),
        tenant_id=_TENANT,
    )
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a2)
    try:
        journal = build_unified_run_journal(_persisted_run(run_id=run_id), runtime_store=store)
        assert journal[0].attempt_id == attempt_a1
        assert journal[0].attempt_id != attempt_a2
    finally:
        reset_active_execution_identity(token)


def test_unified_journal_source_has_no_dynamic_identity_adapters() -> None:
    source = _JOURNAL_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "_TaskLike",
        "_TaskView",
        'type("Ctx"',
        "# type: ignore",
        "task_id or run_id",
        "run_id or task_id",
        'or "default"',
        "tags.get(\"task_id\")",
        "Dict[str, Any]",
        "Mapping[str, Any]",
        "mint_attempt_id",
        "peek_active_execution_identity",
        "bridge_persisted_trace_events",
    ):
        assert forbidden not in source, forbidden
