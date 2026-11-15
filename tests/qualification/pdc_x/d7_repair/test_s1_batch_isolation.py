# © Artur Czarnecki. All rights reserved.

"""F-D7-01 — one failing resume must not abort or strand the rest of its claimed batch."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import (
    T0,
    RecordingExecutor,
    ledger_status,
    save_paused,
    schedule_one_shot,
)

pytestmark = [pytest.mark.unit]

NOW = T0 + timedelta(hours=1)


def _build(tmp_path, executor: RecordingExecutor):
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s1.db")
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,  # type: ignore[arg-type]
        owner_id="sched-s1",
    )
    return scheduler, store


async def test_failing_first_resume_does_not_strand_sibling_claims_of_other_tenants(tmp_path) -> None:
    executor = RecordingExecutor()
    scheduler, store = _build(tmp_path, executor)
    ids: dict[str, str] = {}
    for index, tenant in enumerate(("tenant-a", "tenant-b", "tenant-c")):
        checkpoint = save_paused(store, tenant_id=tenant)
        entry = schedule_one_shot(store, checkpoint, run_at=T0 + timedelta(minutes=index))
        ids[tenant] = entry.schedule_id
    executor.fail_for = frozenset({ids["tenant-a"]})  # the earliest claim of the batch fails

    # The failure is still reported to the caller, but only after the whole batch was processed.
    with pytest.raises(RuntimeError, match="resume failed"):
        await scheduler.tick(now=NOW)

    assert sorted(executor.fired) == sorted([ids["tenant-b"], ids["tenant-c"]])
    status = ledger_status(tmp_path / "s1.db")
    assert status[ids["tenant-b"]] == "completed"
    assert status[ids["tenant-c"]] == "completed"
    # Negative: the failed claim is never silently completed (at-most-once is kept).
    assert status[ids["tenant-a"]] != "completed"
    assert status[ids["tenant-a"]] != "pending"


async def test_stale_claim_at_completion_does_not_abort_the_batch(tmp_path) -> None:
    executor = RecordingExecutor()
    scheduler, store = _build(tmp_path, executor)
    first = schedule_one_shot(store, save_paused(store), run_at=T0)
    second = schedule_one_shot(store, save_paused(store), run_at=T0 + timedelta(minutes=1))

    def lose_the_lease(schedule_id: str, _checkpoint) -> None:
        if schedule_id == first.schedule_id:
            with sqlite3.connect(tmp_path / "s1.db") as conn:
                conn.execute(
                    "UPDATE scheduled_resumes SET status = 'uncertain' WHERE schedule_id = ?",
                    (schedule_id,),
                )

    executor.on_fire = lose_the_lease
    fired = await scheduler.tick(now=NOW)  # complete_claim of the first raises StaleClaimError

    assert second.schedule_id in executor.fired
    status = ledger_status(tmp_path / "s1.db")
    assert status[second.schedule_id] == "completed"
    assert status[first.schedule_id] == "uncertain"  # not overwritten by the stale owner
    assert fired >= 1


async def test_healthy_batch_still_completes_every_claim(tmp_path) -> None:
    executor = RecordingExecutor()
    scheduler, store = _build(tmp_path, executor)
    entries = [schedule_one_shot(store, save_paused(store), run_at=T0 + timedelta(minutes=i)) for i in range(3)]
    assert await scheduler.tick(now=NOW) == 3
    assert {e.schedule_id for e in entries} == set(executor.fired)
    assert set(ledger_status(tmp_path / "s1.db").values()) == {"completed"}


async def test_cancellation_still_propagates_out_of_the_batch(tmp_path) -> None:
    import asyncio

    class _Cancelling(RecordingExecutor):
        async def resume_task(self, task, *, checkpoint):
            raise asyncio.CancelledError

    scheduler, store = _build(tmp_path, _Cancelling())
    schedule_one_shot(store, save_paused(store), run_at=T0)
    with pytest.raises(asyncio.CancelledError):
        await scheduler.tick(now=NOW)
