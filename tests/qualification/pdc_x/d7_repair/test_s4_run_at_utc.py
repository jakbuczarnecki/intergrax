# © Artur Czarnecki. All rights reserved.

"""F-D7-10 — ``run_at_utc`` is validated and normalized to UTC; ``now`` is converted to UTC."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import (
    RESUME_TOKEN,
    RecordingExecutor,
    ledger_status,
    save_paused,
)

pytestmark = [pytest.mark.unit]

NOW = datetime(2026, 12, 1, 8, 30, tzinfo=UTC)


def _entry(checkpoint: TaskCheckpoint, run_at: str, schedule_id: str) -> ScheduledResume:
    return ScheduledResume(
        schedule_id=schedule_id,
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=run_at,
    )


def test_offset_timestamps_are_normalized_to_utc() -> None:
    offset = ScheduledResume(
        task_id="t", tenant_id="x", resume_token=RESUME_TOKEN, run_at_utc="2026-12-01T05:00:00-05:00"
    )
    assert offset.run_at_utc == "2026-12-01T10:00:00+00:00"
    zulu = ScheduledResume(task_id="t", tenant_id="x", resume_token=RESUME_TOKEN, run_at_utc="2026-12-01T10:00:00Z")
    assert zulu.run_at_utc == "2026-12-01T10:00:00+00:00"


@pytest.mark.parametrize("bad", ["", "not-a-time", "2026-12-01T10:00:00", "2026-13-40T00:00:00+00:00", "  "])
def test_unparseable_or_naive_run_at_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        ScheduledResume(task_id="t", tenant_id="x", resume_token=RESUME_TOKEN, run_at_utc=bad)


async def test_offset_run_at_fires_by_its_utc_instant_not_its_text(tmp_path) -> None:
    db = tmp_path / "s4.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    executor = RecordingExecutor()
    scheduler = LongRunningScheduler(store, executor, schedule_store=store, ledger=store, owner_id="s4")  # type: ignore[arg-type]
    not_yet = store.schedule(_entry(save_paused(store), "2026-12-01T05:00:00-05:00", "later-10z"))  # 10:00Z
    due = store.schedule(_entry(save_paused(store), "2026-12-01T10:00:00+02:00", "due-08z"))  # 08:00Z

    await scheduler.tick(now=NOW)  # 08:30Z

    status = ledger_status(db)
    assert status[due.schedule_id] == "completed"
    assert status[not_yet.schedule_id] == "pending"  # fired 90 minutes early before the repair
    assert executor.fired == [due.schedule_id]


async def test_non_utc_now_is_converted_before_the_comparison(tmp_path) -> None:
    db = tmp_path / "s4b.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    scheduler = LongRunningScheduler(store, RecordingExecutor(), schedule_store=store, ledger=store, owner_id="s4b")  # type: ignore[arg-type]
    entry = store.schedule(_entry(save_paused(store), "2026-12-01T09:00:00+00:00", "at-09z"))
    plus_one = timezone(timedelta(hours=1))
    await scheduler.tick(now=datetime(2026, 12, 1, 9, 30, tzinfo=plus_one))  # = 08:30Z, not due yet
    assert ledger_status(db)[entry.schedule_id] == "pending"
    await scheduler.tick(now=datetime(2026, 12, 1, 10, 30, tzinfo=plus_one))  # = 09:30Z, due
    assert ledger_status(db)[entry.schedule_id] == "completed"


def test_store_normalizes_and_rejects_its_cutoff_argument(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s4c.db")
    entry = store.schedule(_entry(save_paused(store), "2026-12-01T10:00:00+00:00", "cutoff"))
    due = store.list_due(before_utc_iso="2026-12-01T05:30:00-05:00")  # = 10:30Z
    assert [row.schedule_id for row in due] == [entry.schedule_id]
    assert store.list_due(before_utc_iso="2026-12-01T04:30:00-05:00") == []  # = 09:30Z
    with pytest.raises(ValueError):
        store.list_due(before_utc_iso="not-a-time")
    with pytest.raises(ValueError):
        store.claim_due(before_utc_iso="", owner_id="o", lease_seconds=5)


def test_malformed_legacy_row_does_not_poison_claim_due(tmp_path) -> None:
    db = tmp_path / "s4d.db"
    store = SQLiteTaskCheckpointStore(db_path=db)
    good = store.schedule(_entry(save_paused(store), "2026-12-01T08:00:00+00:00", "good"))
    with sqlite3.connect(db) as conn:  # a legacy row that predates validation
        conn.execute(
            "INSERT INTO scheduled_resumes (schedule_id, task_id, tenant_id, resume_token, run_at_utc, status,"
            " resume_metadata_json, created_at_utc)"
            " VALUES ('legacy-bad', 't', 'x', 'tok', '', 'pending', '{}', '2020-01-01T00:00:00+00:00')"
        )
    claims = store.claim_due(before_utc_iso=NOW.isoformat(), owner_id="o", lease_seconds=60)
    assert [claim.schedule_id for claim in claims] == [good.schedule_id]
    assert ledger_status(db)["legacy-bad"] == "uncertain"  # quarantined, not retried forever
    assert store.list_due(before_utc_iso=NOW.isoformat()) == []
