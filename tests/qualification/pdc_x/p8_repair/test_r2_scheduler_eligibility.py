# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R2 (G-02): a failure in the eligibility phase never ends a recurrence chain."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from intergrax.contracts.execution_terminal import ExecutionTerminalError, ExecutionTerminalOutcome
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import (
    FAR_FUTURE,
    T0,
    RecordingExecutor,
    ledger_status,
    save_paused,
    schedule_one_shot,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FlakyStore(SQLiteTaskCheckpointStore):
    """Checkpoint store whose next ``get_by_token`` / ``get_latest`` raises once, on demand."""

    def __init__(self, *, db_path: Path) -> None:
        super().__init__(db_path=db_path)
        self.fail_get_by_token = False
        self.fail_get_latest = False

    def get_by_token(self, task_id, tenant_id, resume_token):
        if self.fail_get_by_token:
            self.fail_get_by_token = False
            raise RuntimeError("transient checkpoint read error")
        return super().get_by_token(task_id, tenant_id, resume_token)

    def get_latest(self, task_id, tenant_id):
        if self.fail_get_latest:
            self.fail_get_latest = False
            raise RuntimeError("transient latest-checkpoint read error")
        return super().get_latest(task_id, tenant_id)


class _TerminalAuthority:
    """Terminal service double: unavailable, or reporting a terminal outcome, on demand."""

    def __init__(self) -> None:
        self.unavailable = False
        self.outcome: ExecutionTerminalOutcome | None = None

    def get_terminal_record(self, *, tenant_id: str, task_id: str):
        if self.unavailable:
            raise ExecutionTerminalError("terminal authority unavailable")
        if self.outcome is None:
            return None
        return SimpleNamespace(outcome=self.outcome)


def _rule(checkpoint) -> RecurrenceRule:
    return RecurrenceRule(
        rule_id="rule-r2",
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        resume_token=checkpoint.resume_token,
        interval_seconds=60,
        start_utc=T0.isoformat(),
    )


def _pending(store: SQLiteTaskCheckpointStore) -> list[str]:
    return [row.run_at_utc for row in store.list_due(before_utc_iso=FAR_FUTURE)]


def _build(store, executor, *, terminal: _TerminalAuthority | None = None, recurring: bool = True):
    return LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,  # type: ignore[arg-type]
        owner_id="sched-r2",
        recurrence=RecurrenceMaterializer(store) if recurring else None,
        execution_terminal=terminal,  # type: ignore[arg-type]
    )


def _recurring_setup(tmp_path: Path):
    db = tmp_path / "r2.db"
    store = _FlakyStore(db_path=db)
    checkpoint = save_paused(store)
    assert RecurrenceMaterializer(store).start(_rule(checkpoint)) is not None
    return db, store


async def _assert_released_then_retried(db, store, scheduler, executor) -> None:
    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0  # nothing executed, claim handed back
    assert executor.fired == []
    assert list(ledger_status(db).values()) == ["pending"]  # released: neither running nor skipped

    assert await scheduler.tick(now=T0 + timedelta(seconds=2)) == 1  # retried at the next healthy tick
    assert len(executor.fired) == 1
    assert _pending(store) == [(T0 + timedelta(seconds=60)).isoformat()]  # exactly one successor
    assert sorted(ledger_status(db).values()) == ["completed", "pending"]


async def test_get_by_token_failure_releases_the_claim_and_keeps_the_chain(tmp_path: Path) -> None:
    db, store = _recurring_setup(tmp_path)
    executor = RecordingExecutor()
    store.fail_get_by_token = True
    await _assert_released_then_retried(db, store, _build(store, executor), executor)


async def test_get_latest_failure_in_the_materialization_gate_keeps_the_chain(tmp_path: Path) -> None:
    db, store = _recurring_setup(tmp_path)
    executor = RecordingExecutor()
    store.fail_get_latest = True
    await _assert_released_then_retried(db, store, _build(store, executor), executor)


async def test_unavailable_terminal_authority_is_not_reported_as_not_resumable(tmp_path: Path) -> None:
    db, store = _recurring_setup(tmp_path)
    executor = RecordingExecutor()
    terminal = _TerminalAuthority()
    terminal.unavailable = True
    scheduler = _build(store, executor, terminal=terminal)

    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0
    assert list(ledger_status(db).values()) == ["pending"]  # not "skipped": the chain is still alive
    assert _pending(store) == [T0.isoformat()]

    terminal.unavailable = False
    assert await scheduler.tick(now=T0 + timedelta(seconds=2)) == 1
    assert len(executor.fired) == 1
    assert _pending(store) == [(T0 + timedelta(seconds=60)).isoformat()]


async def test_a_real_terminal_outcome_still_ends_the_chain_skipped(tmp_path: Path) -> None:
    db, store = _recurring_setup(tmp_path)
    terminal = _TerminalAuthority()
    terminal.outcome = ExecutionTerminalOutcome.CANCELLED
    scheduler = _build(store, RecordingExecutor(), terminal=terminal)

    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0
    assert list(ledger_status(db).values()) == ["skipped"]
    assert _pending(store) == []


async def test_a_one_shot_entry_keeps_the_lease_behaviour_when_the_read_fails(tmp_path: Path) -> None:
    db = tmp_path / "oneshot.db"
    store = _FlakyStore(db_path=db)
    checkpoint = save_paused(store)
    entry = schedule_one_shot(store, checkpoint, run_at=T0)
    store.fail_get_by_token = True
    scheduler = _build(store, RecordingExecutor(), recurring=False)

    with pytest.raises(RuntimeError, match="transient checkpoint read error"):
        await scheduler.tick(now=T0 + timedelta(seconds=1))
    assert ledger_status(db)[entry.schedule_id] == "running"  # left to the lease, not released
