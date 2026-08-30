# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    as_of_boundary_for_positioned,
)
from intergrax.runtime.events.persistence_contract import NullRuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunStats,
)

pytestmark = [pytest.mark.gate, pytest.mark.no_ci]

_TENANT = "tenant-a"


def _event(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    execution_id: str | None = None,
    event_id: str | None = None,
    timestamp: datetime | None = None,
    event_type: RuntimeEventType = RuntimeEventType.STEP_STARTED,
) -> RuntimeEvent:
    identity = runtime_event_test_identity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=_TENANT,
        **identity,
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=timestamp or datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        correlation_id=run_id,
    )


def _persisted_run(run_id: str) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=_TENANT,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=[],
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_execution_position_ordering_ignores_wall_clock(
    backend: str,
    tmp_path: Path,
) -> None:
    if backend == "memory":
        store: InMemoryRuntimeEventStore | SQLiteRuntimeEventStore = InMemoryRuntimeEventStore()
    else:
        store = SQLiteRuntimeEventStore(db_path=tmp_path / "events.db")
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    earlier_ts = datetime(2026, 6, 7, 9, 59, 59, tzinfo=timezone.utc)
    later_ts = datetime(2026, 6, 7, 10, 0, 5, tzinfo=timezone.utc)
    first = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=later_ts,
    )
    second = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=earlier_ts,
    )
    p1 = store.append(first, tenant_id=_TENANT)
    p2 = store.append(second, tenant_id=_TENANT)
    assert p1.position.value < p2.position.value


@pytest.mark.unit
def test_equal_timestamps_receive_distinct_positions() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    same_ts = datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc)
    p1 = store.append(
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id, timestamp=same_ts),
        tenant_id=_TENANT,
    )
    p2 = store.append(
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id, timestamp=same_ts),
        tenant_id=_TENANT,
    )
    assert p1.position != p2.position


@pytest.mark.unit
def test_execution_position_is_run_scoped() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    attempt_id = mint_attempt_id()
    p_a = store.append(
        _event(task_id=task_id, run_id=run_a, attempt_id=attempt_id),
        tenant_id=_TENANT,
    )
    p_b = store.append(
        _event(task_id=task_id, run_id=run_b, attempt_id=attempt_id),
        tenant_id=_TENANT,
    )
    assert p_a.position == ExecutionEventPosition(1)
    assert p_b.position == ExecutionEventPosition(1)


@pytest.mark.unit
def test_retry_and_resume_preserve_run_position_sequence() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    positions = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a2,
                event_type=RuntimeEventType.RETRY_STARTED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.RESUMED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    assert [position.position.value for position in positions] == [1, 2, 3, 4]


@pytest.mark.unit
def test_idempotent_append_preserves_position() -> None:
    store = InMemoryRuntimeEventStore()
    event = _event(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    first = store.append(event, tenant_id=_TENANT)
    second = store.append(event, tenant_id=_TENANT)
    assert first.position == second.position
    assert len(store.list_positioned_for_run(event.run_id, tenant_id=_TENANT)) == 1


@pytest.mark.unit
def test_different_event_ids_cannot_share_position() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    p1 = store.append(
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id),
        tenant_id=_TENANT,
    )
    p2 = store.append(
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id),
        tenant_id=_TENANT,
    )
    assert p1.position != p2.position


@pytest.mark.unit
def test_as_of_boundary_validation_and_prefix() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = [
        store.append(_event(task_id=task_id, run_id=run_id, attempt_id=attempt_id), tenant_id=_TENANT)
        for _ in range(3)
    ]
    boundary = as_of_boundary_for_positioned(positioned[1])
    prefix = store.list_positioned_through(boundary, tenant_id=_TENANT)
    assert [row.event_id for row in prefix] == [
        positioned[0].event_id,
        positioned[1].event_id,
    ]
    with pytest.raises(ValueError, match="run_id does not match"):
        boundary.includes(
            store.append(
                _event(
                    task_id=task_id,
                    run_id=mint_run_id(),
                    attempt_id=attempt_id,
                ),
                tenant_id=_TENANT,
            )
        )


@pytest.mark.unit
def test_as_of_boundary_is_inclusive_and_timestamp_independent() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    first = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            timestamp=datetime(2026, 6, 7, 10, 0, 5, tzinfo=timezone.utc),
        ),
        tenant_id=_TENANT,
    )
    second = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            timestamp=datetime(2026, 6, 7, 9, 0, 0, tzinfo=timezone.utc),
        ),
        tenant_id=_TENANT,
    )
    boundary = AsOfBoundary(run_id=run_id, position=first.position)
    prefix = store.list_positioned_through(boundary, tenant_id=_TENANT)
    assert [row.event_id for row in prefix] == [first.event_id]
    assert boundary.includes(second) is False


@pytest.mark.unit
def test_unified_journal_uses_execution_position_not_timestamp() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    first = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=datetime(2026, 6, 7, 10, 0, 5, tzinfo=timezone.utc),
    )
    second = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        timestamp=datetime(2026, 6, 7, 9, 0, 0, tzinfo=timezone.utc),
    )
    store.append(first, tenant_id=_TENANT)
    store.append(second, tenant_id=_TENANT)
    journal = build_unified_run_journal(_persisted_run(run_id), runtime_store=store)
    assert [event.event_id for event in journal] == [first.event_id, second.event_id]


@pytest.mark.unit
def test_concurrent_append_allocates_distinct_positions(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "concurrent.db")
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    events = [
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
        for _ in range(8)
    ]

    def _append(event: RuntimeEvent) -> int:
        return store.append(event, tenant_id=_TENANT).position.value

    with ThreadPoolExecutor(max_workers=4) as pool:
        positions = list(pool.map(_append, events))
    assert len(positions) == len(set(positions))
    assert sorted(positions) == list(range(1, len(events) + 1))


@pytest.mark.unit
def test_null_runtime_event_persistence_returns_positions() -> None:
    store = NullRuntimeEventPersistence()
    event = _event(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    first = store.append(event, tenant_id=_TENANT)
    second = store.append(event, tenant_id=_TENANT)
    assert first.position == second.position == ExecutionEventPosition(1)
