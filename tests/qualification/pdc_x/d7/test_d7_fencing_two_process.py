# © Artur Czarnecki. All rights reserved.

"""PDC-D7-R012 — lease/fencing proven across OS processes over a shared SQLite ledger.

Two independent processes run the real ``LongRunningScheduler`` against one on-disk
``SQLiteTaskCheckpointStore``. A file barrier releases both at once to maximize
contention. All waits are bounded and children are killed on timeout.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.long_running.recurrence import (
    MisfirePolicy,
    RecurrenceMaterializer,
    RecurrenceRule,
    occurrence_schedule_id,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import (
    RESUME_TOKEN,
    TENANT_ID,
    paused_checkpoint,
)

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKER = Path(__file__).with_name("_fencing_worker.py")
_PROCESS_TIMEOUT_SECONDS = 90.0
_BARRIER_TIMEOUT_SECONDS = 60.0
T0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)


def _spawn(
    owner: str,
    *,
    db: Path,
    sync_dir: Path,
    fire_log: Path,
    now: datetime,
    mode: str = "tick",
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            str(_WORKER),
            "--repo-root",
            str(_REPO_ROOT),
            "--db",
            str(db),
            "--owner",
            owner,
            "--sync-dir",
            str(sync_dir),
            "--fire-log",
            str(fire_log),
            "--now",
            now.isoformat(),
            "--mode",
            mode,
            "--deadline-seconds",
            "60",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _await_ready(sync_dir: Path, owners: list[str]) -> None:
    deadline = time.monotonic() + _BARRIER_TIMEOUT_SECONDS
    while not all((sync_dir / f"ready.{owner}").exists() for owner in owners):
        if time.monotonic() > deadline:
            raise AssertionError("child processes did not reach the start barrier")
        time.sleep(0.02)


def _finish(procs: list[subprocess.Popen]) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    try:
        for proc in procs:
            try:
                out, err = proc.communicate(timeout=_PROCESS_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                raise AssertionError(f"child timed out; stderr={err!r}") from None
            results.append((proc.returncode, out, err))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    return results


def _fires(fire_log: Path) -> list[tuple[str, str]]:
    if not fire_log.exists():
        return []
    rows = [line.split("\t") for line in fire_log.read_text(encoding="utf-8").splitlines() if line]
    return [(schedule_id, owner) for schedule_id, owner in rows]


def _ledger_rows(db: Path) -> dict[str, tuple[str, int]]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT schedule_id, status, fence FROM scheduled_resumes").fetchall()
    return {schedule_id: (status, int(fence)) for schedule_id, status, fence in rows}


def test_two_processes_fire_each_recurring_occurrence_exactly_once(tmp_path) -> None:
    db = tmp_path / "ledger.db"
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    fire_log = tmp_path / "fires.log"
    store = SQLiteTaskCheckpointStore(db_path=db)
    store.save(paused_checkpoint())
    rule = RecurrenceRule(
        rule_id="rule-fence",
        tenant_id=TENANT_ID,
        task_id=store.list_paused()[0].task_id,
        resume_token=RESUME_TOKEN,
        interval_seconds=60,
        start_utc=T0.isoformat(),
        misfire=MisfirePolicy.catch_up(10),
    )
    assert RecurrenceMaterializer(store).start(rule) is not None

    now = T0 + timedelta(seconds=200)  # occurrences +0, +60, +120, +180 are due
    owners = ["proc-a", "proc-b"]
    procs = [_spawn(owner, db=db, sync_dir=sync_dir, fire_log=fire_log, now=now) for owner in owners]
    try:
        _await_ready(sync_dir, owners)
        (sync_dir / "go").write_text("go", encoding="utf-8")
        results = _finish(procs)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    for returncode, out, err in results:
        assert returncode == 0, f"child failed: out={out!r} err={err!r}"

    expected = [occurrence_schedule_id(rule.rule_id, T0 + timedelta(seconds=60 * k)) for k in range(4)]
    fired_ids = [schedule_id for schedule_id, _ in _fires(fire_log)]
    assert sorted(fired_ids) == sorted(expected)  # every due occurrence fired ...
    assert len(fired_ids) == len(set(fired_ids))  # ... exactly once across both processes

    rows = _ledger_rows(db)
    for schedule_id in expected:
        assert rows[schedule_id] == ("completed", 1)  # one claim generation each, no re-claim
    next_id = occurrence_schedule_id(rule.rule_id, T0 + timedelta(seconds=240))
    assert rows[next_id][0] == "pending"  # the chain continues with exactly one future one-shot
    assert len(rows) == 5


def test_crash_after_claim_is_never_refired_and_stale_completion_is_rejected(tmp_path) -> None:
    db = tmp_path / "ledger.db"
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    fire_log = tmp_path / "fires.log"
    store = SQLiteTaskCheckpointStore(db_path=db)
    checkpoint = paused_checkpoint()
    store.save(checkpoint)
    due = T0 - timedelta(minutes=5)
    entry = store.schedule(
        ScheduledResume(
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            run_at_utc=due.isoformat(),
        )
    )

    # Process A claims the occurrence and dies before completing or firing it.
    (sync_dir / "go").write_text("go", encoding="utf-8")
    crasher = _spawn("proc-crash", db=db, sync_dir=sync_dir, fire_log=fire_log, now=T0, mode="claim_and_die")
    ((code, out, _),) = _finish([crasher])
    assert code == 3 and "claimed=1" in out
    assert _ledger_rows(db)[entry.schedule_id] == ("running", 1)

    # Process B runs the scheduler: the orphaned claim is NOT re-fired (at-most-once).
    survivor = _spawn("proc-survivor", db=db, sync_dir=sync_dir, fire_log=fire_log, now=T0)
    ((code, out, err),) = _finish([survivor])
    assert code == 0, err
    assert _fires(fire_log) == []
    assert _ledger_rows(db)[entry.schedule_id] == ("uncertain", 1)

    # The dead owner's late completion (stale fence) is rejected by the ledger.
    stale_claim = ScheduledResumeClaim(
        schedule_id=entry.schedule_id,
        owner_id="proc-crash",
        lease_expires_at=T0,
        fence=1,
        entry=entry,
    )
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale_claim)
    assert _ledger_rows(db)[entry.schedule_id] == ("uncertain", 1)
