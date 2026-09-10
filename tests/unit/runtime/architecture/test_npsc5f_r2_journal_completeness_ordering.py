# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R2 — journal completeness and run-local ordering contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import RunId, mint_run_id, mint_task_id
from intergrax.runtime.events.execution_position import AsOfBoundary, ExecutionEventPosition
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import (
    JournalCursorScopeMismatchError,
    JournalReadLimitExceededError,
    PositionedJournalBoundaryNotFoundError,
    PositionedJournalPrefixTruncatedError,
    RunJournalContinuationCursor,
    build_unified_run_journal,
    load_complete_run_journal,
    load_positioned_run_journal_through,
    read_run_journal_page,
)
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-r2"
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _persisted_run(run_id: str, tenant_id: str = _TENANT) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
        events=[],
    )


def _append_n(
    store: RuntimeEventPersistence,
    *,
    run_id: RunId,
    count: int,
    tenant_id: str = _TENANT,
) -> None:
    task_id = mint_task_id()
    for _ in range(count):
        store.append(
            sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id),
            tenant_id=tenant_id,
        )


@pytest.mark.parametrize(
    ("label", "store"),
    [
        ("memory", InMemoryRuntimeEventStore()),
    ],
)
def test_r2_complete_loader_returns_all_events_not_truncated(label: str, store: RuntimeEventPersistence) -> None:
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=8)
    journal = load_complete_run_journal(
        store,
        tenant_id=_TENANT,
        run_id=run_id,
        max_events=100,
        page_size=3,
    )
    assert len(journal) == 8


def test_r2_page_exact_limit_proves_complete(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "exact.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=3)
    page = read_run_journal_page(
        store,
        tenant_id=_TENANT,
        run_id=run_id,
        page_size=3,
    )
    assert len(page.events) == 3
    assert page.is_complete is True
    assert page.next_cursor is None
    store.close()


def test_r2_page_limit_plus_one_exposes_continuation(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "nplus1.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=4)
    page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=3)
    assert len(page.events) == 3
    assert page.is_complete is False
    assert page.next_cursor is not None
    store.close()


def test_r2_empty_run_complete(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=3)
    assert page.events == ()
    assert page.is_complete is True


def test_r2_multi_page_concatenation(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "multipage.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=8)
    collected: list[str] = []
    cursor = None
    while True:
        page = read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=3,
            cursor=cursor,
        )
        collected.extend(event.event_id for event in page.events)
        if page.is_complete:
            break
        cursor = page.next_cursor
    full = load_complete_run_journal(
        store,
        tenant_id=_TENANT,
        run_id=run_id,
        max_events=100,
        page_size=100,
    )
    assert collected == [event.event_id for event in full]
    store.close()


def test_r2_wrong_run_cursor_blocked(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    run_a = mint_run_id()
    run_b = mint_run_id()
    _append_n(store, run_id=run_a, count=2)
    first = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_a, page_size=1)
    assert first.next_cursor is not None
    cursor = RunJournalContinuationCursor(
        tenant_id=_TENANT,
        run_id=run_b,
        exclusive_after=first.next_cursor.exclusive_after,
        snapshot_through=first.next_cursor.snapshot_through,
    )
    with pytest.raises(JournalCursorScopeMismatchError):
        read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_a,
            page_size=1,
            cursor=cursor,
        )


def test_r2_wrong_tenant_cursor_blocked(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=2)
    first = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=1)
    assert first.next_cursor is not None
    cursor = RunJournalContinuationCursor(
        tenant_id="other-tenant",
        run_id=run_id,
        exclusive_after=first.next_cursor.exclusive_after,
        snapshot_through=first.next_cursor.snapshot_through,
    )
    with pytest.raises(JournalCursorScopeMismatchError):
        read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=1,
            cursor=cursor,
        )


def test_r2_complete_loader_max_events_fail_closed(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=5)
    with pytest.raises(JournalReadLimitExceededError):
        load_complete_run_journal(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            max_events=3,
            page_size=2,
        )


def test_r2_concurrent_append_snapshot_excludes_new_events() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=3)
    first_page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=1)
    assert len(first_page.events) == 1
    assert first_page.next_cursor is not None
    _append_n(store, run_id=run_id, count=2)
    tail: list[str] = []
    cursor = first_page.next_cursor
    while cursor is not None:
        page = read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=10,
            cursor=cursor,
        )
        tail.extend(event.event_id for event in page.events)
        if page.is_complete:
            break
        cursor = page.next_cursor
    assert len(tail) == 2
    full_snapshot = [first_page.events[0].event_id, *tail]
    assert len(full_snapshot) == 3


def test_r2_run_local_positions_two_runs(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "tworuns.db")
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    pos_a = store.append(
        sample_runtime_event(tenant_id=_TENANT, task_id=task_id, run_id=run_a),
        tenant_id=_TENANT,
    ).position
    pos_b = store.append(
        sample_runtime_event(tenant_id=_TENANT, task_id=task_id, run_id=run_b),
        tenant_id=_TENANT,
    ).position
    assert pos_a.value == pos_b.value == 1
    store.close()


def test_r2_sqlite_list_for_task_not_position_only_order() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "events" / "stores" / "sqlite_runtime_event_store.py"
    ).read_text(encoding="utf-8")
    assert "ORDER BY run_id ASC, execution_position ASC" in source
    assert "ORDER BY execution_position ASC\n                LIMIT" not in source


def test_r2_task_grouped_by_run_ordering(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    for _ in range(2):
        store.append(
            sample_runtime_event(tenant_id=_TENANT, task_id=task_id, run_id=run_a),
            tenant_id=_TENANT,
        )
    for _ in range(2):
        store.append(
            sample_runtime_event(tenant_id=_TENANT, task_id=task_id, run_id=run_b),
            tenant_id=_TENANT,
        )
    grouped = store.list_positioned_for_task_grouped_by_run(task_id, tenant_id=_TENANT, limit=10)
    assert len(grouped.runs) == 2
    for _, rows in grouped.runs:
        positions = [row.position.value for row in rows]
        assert positions == sorted(positions)


def test_r2_prefix_helper_still_works(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "prefix.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=4)
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(2))
    prefix = load_positioned_run_journal_through(
        store,
        tenant_id=_TENANT,
        boundary=boundary,
        initial_limit=1,
        max_limit=100,
    )
    assert [row.position.value for row in prefix] == [1, 2]
    store.close()


def test_r2_prefix_truncation_still_fail_closed(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "prefix_trunc.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=10)
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(10))
    with pytest.raises(PositionedJournalPrefixTruncatedError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
            initial_limit=2,
            max_limit=4,
        )
    store.close()


def test_r2_prefix_missing_boundary_still_fail_closed(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "prefix_missing.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=2)
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(5))
    with pytest.raises(PositionedJournalBoundaryNotFoundError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
            initial_limit=10,
            max_limit=100,
        )
    store.close()


def test_r2_build_unified_run_journal_complete(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "unified.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=6)
    journal = build_unified_run_journal(
        _persisted_run(run_id),
        runtime_store=store,
        page_size=2,
    )
    assert len(journal) == 6
    store.close()


@pytest.mark.parametrize("label,store", [("memory", InMemoryRuntimeEventStore())])
def test_r2_store_parity_page_semantics(label: str, store: RuntimeEventPersistence) -> None:
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=5)
    page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=2)
    assert len(page.events) == 2
    assert page.is_complete is False
