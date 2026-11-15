# © Artur Czarnecki. All rights reserved.

"""F-D7-02 / F-D7-13 — the recurrence chain survives crashes and ends explicitly."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from intergrax.runtime.long_running.recurrence import (
    RECURRENCE_OCCURRENCE_METADATA_KEY,
    RecurrenceMaterializer,
    RecurrenceRule,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import (
    FAR_FUTURE,
    T0,
    RecordingExecutor,
    ledger_status,
    save_paused,
)

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER = Path(__file__).with_name("_crash_worker.py")


def _rule(checkpoint, **overrides) -> RecurrenceRule:
    params = {
        "rule_id": "rule-s2",
        "tenant_id": checkpoint.tenant_id,
        "task_id": checkpoint.task_id,
        "resume_token": checkpoint.resume_token,
        "interval_seconds": 60,
        "start_utc": T0.isoformat(),
    }
    params.update(overrides)
    return RecurrenceRule(**params)


def _pending(store: SQLiteTaskCheckpointStore) -> list[str]:
    return [row.run_at_utc for row in store.list_due(before_utc_iso=FAR_FUTURE)]


def _build(executor, store: SQLiteTaskCheckpointStore):
    return LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,  # type: ignore[arg-type]
        owner_id="sched-s2",
        recurrence=RecurrenceMaterializer(store),
    )


def _run_dying_scheduler(db: Path) -> int:
    proc = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--repo-root",
            str(_REPO_ROOT),
            "--db",
            str(db),
            "--now",
            (T0 + timedelta(seconds=1)).isoformat(),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode


def test_successor_exists_after_the_process_dies_while_the_occurrence_runs(tmp_path) -> None:
    db = tmp_path / "crash.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    assert RecurrenceMaterializer(store).start(_rule(checkpoint)) is not None

    assert _run_dying_scheduler(db) == 3

    # Occurrence 0 is still RUNNING (killed) and its successor is already pending.
    assert sorted(ledger_status(db).values()) == ["pending", "running"]
    assert _pending(store) == [(T0 + timedelta(seconds=60)).isoformat()]


async def test_chain_continues_after_a_crash_when_the_survivor_ticks(tmp_path) -> None:
    db = tmp_path / "crash.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    RecurrenceMaterializer(store).start(_rule(checkpoint))
    assert _run_dying_scheduler(db) == 3

    executor = RecordingExecutor()
    scheduler = _build(executor, store)
    assert await scheduler.tick(now=T0 + timedelta(seconds=61)) == 1
    assert len(executor.fired) == 1
    assert _pending(store) == [(T0 + timedelta(seconds=120)).isoformat()]


class _FlakyStore(SQLiteTaskCheckpointStore):
    def __init__(self, *, db_path) -> None:
        super().__init__(db_path=db_path)
        self.fail_next_schedule = False

    def schedule(self, entry):
        if self.fail_next_schedule:
            self.fail_next_schedule = False
            raise RuntimeError("transient store error")
        return super().schedule(entry)


async def test_transient_store_error_while_materializing_releases_the_claim_and_retries(tmp_path) -> None:
    db = tmp_path / "flaky.db"
    store = _FlakyStore(db_path=db)
    checkpoint = save_paused(store)
    RecurrenceMaterializer(store).start(_rule(checkpoint))
    executor = RecordingExecutor()
    scheduler = _build(executor, store)

    store.fail_next_schedule = True
    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0  # nothing executed, claim handed back
    assert executor.fired == []
    assert list(ledger_status(db).values()) == ["pending"]

    assert await scheduler.tick(now=T0 + timedelta(seconds=2)) == 1  # retried at the next tick
    assert len(executor.fired) == 1
    assert _pending(store) == [(T0 + timedelta(seconds=60)).isoformat()]  # exactly one successor
    assert sorted(ledger_status(db).values()) == ["completed", "pending"]


async def test_failing_resume_does_not_end_the_chain(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "fail.db")
    checkpoint = save_paused(store)
    first = RecurrenceMaterializer(store).start(_rule(checkpoint))
    assert first is not None
    scheduler = _build(RecordingExecutor(fail_for=frozenset({first.schedule_id})), store)
    with pytest.raises(RuntimeError, match="resume failed"):
        await scheduler.tick(now=T0 + timedelta(seconds=1))
    assert _pending(store) == [(T0 + timedelta(seconds=60)).isoformat()]


async def test_superseded_target_ends_the_chain_with_one_skipped_row(tmp_path) -> None:
    db = tmp_path / "super.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    RecurrenceMaterializer(store).start(_rule(checkpoint))
    executor = RecordingExecutor()

    def re_pause(_schedule_id, current) -> None:
        newer = current.model_copy(update={"checkpoint_id": "ckpt-newer", "resume_token": "token-newer"})
        store.save(newer, expected_revision=1)

    executor.on_fire = re_pause
    scheduler = _build(executor, store)

    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 1  # occurrence 0 runs and re-pauses
    assert await scheduler.tick(now=T0 + timedelta(seconds=61)) == 0  # occurrence 1 cannot resume
    assert sorted(ledger_status(db).values()) == ["completed", "skipped"]  # skipped, not completed
    assert _pending(store) == []  # the chain ended explicitly: no successor
    assert len(executor.fired) == 1


async def test_missing_target_checkpoint_skips_and_mints_no_successor(tmp_path) -> None:
    db = tmp_path / "missing.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    RecurrenceMaterializer(store).start(_rule(checkpoint, resume_token="no-such-token"))
    scheduler = _build(RecordingExecutor(), store)
    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0
    assert list(ledger_status(db).values()) == ["skipped"]
    assert _pending(store) == []


async def test_one_shot_with_a_missing_target_keeps_the_completed_status(tmp_path) -> None:
    db = tmp_path / "oneshot.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    entry = store.schedule(
        ScheduledResume(
            schedule_id="oneshot-missing",
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token="no-such-token",
            run_at_utc=T0.isoformat(),
        )
    )
    scheduler = _build(RecordingExecutor(), store)
    await scheduler.tick(now=T0 + timedelta(seconds=1))
    assert ledger_status(db)[entry.schedule_id] == "completed"


async def test_invalid_persisted_rule_is_skipped_fail_closed(tmp_path) -> None:
    db = tmp_path / "bad.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = save_paused(store)
    first = RecurrenceMaterializer(store).start(_rule(checkpoint))
    assert first is not None
    with sqlite3.connect(db) as conn:  # corrupt the persisted rule behind the API
        conn.execute(
            "UPDATE scheduled_resumes SET resume_metadata_json = ? WHERE schedule_id = ?",
            (
                '{"recurrence_rule": {"rule_id": 1}, "' + RECURRENCE_OCCURRENCE_METADATA_KEY + '": "x"}',
                first.schedule_id,
            ),
        )
    executor = RecordingExecutor()
    scheduler = _build(executor, store)
    assert await scheduler.tick(now=T0 + timedelta(seconds=1)) == 0
    assert executor.fired == []  # nothing is executed under an invalid rule
    assert list(ledger_status(db).values()) == ["skipped"]
