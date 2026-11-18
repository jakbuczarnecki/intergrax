# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-E NT-RW-REC-06 (PDC-D7-R015) and NT-RW-REC-07 (G-20): UNCERTAIN reconciler, capped release.

REC-06: ``UncertainReconciler`` resolves every UNCERTAIN schedule of its source exactly once with a
recorded result and a recorded double-execution check; only a proven not-started entry is redriven.
The source here is in memory: the SQLite and PostgreSQL schedule stores do not implement the port yet.

REC-07: a permanently failing materialization (a) or eligibility read (b) is released at most
``max_claim_release_attempts`` times, then the claim is skipped with one stable error code.
REC-07c (control, already green before this ticket): ``OverflowError`` goes to ``skip_claim`` at once.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume, ScheduledResumeStatus
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeClaim
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import T0, RecordingExecutor, save_paused

pytestmark = [pytest.mark.unit]

_CAP_CODE = "scheduled_resume_release_cap_reached"


# ---------------------------------------------------------------- REC-06


class _InMemoryUncertainSource:
    """``UncertainScheduleSource`` over a dict; the resolve is fenced on ``ScheduledResume.fence``."""

    def __init__(self, entries: Sequence[ScheduledResume]) -> None:
        self.rows: dict[str, ScheduledResume] = {entry.schedule_id: entry for entry in entries}
        self.resolve_calls: list[tuple[str, str]] = []

    def list_uncertain(self, limit: int) -> Sequence[ScheduledResume]:
        rows = [row for row in self.rows.values() if row.status is ScheduledResumeStatus.UNCERTAIN]
        return rows[:limit]

    def resolve_uncertain(self, schedule_id: str, *, expected_fence: int, outcome: object) -> bool:
        from intergrax.runtime.long_running.uncertain_reconciler import UncertainOutcome

        assert isinstance(outcome, UncertainOutcome)
        self.resolve_calls.append((schedule_id, outcome.value))
        row = self.rows.get(schedule_id)
        if row is None or row.status is not ScheduledResumeStatus.UNCERTAIN or row.fence != expected_fence:
            return False
        if outcome is UncertainOutcome.EXECUTION_PROVEN:
            status = ScheduledResumeStatus.COMPLETED
        elif outcome is UncertainOutcome.NOT_STARTED_PROVEN:
            status = ScheduledResumeStatus.PENDING
        else:
            return False
        self.rows[schedule_id] = row.model_copy(update={"status": status, "fence": row.fence + 1})
        return True


class _Evidence:
    """Terminal authority and attempt reads per task id; ``None`` means the read is not available."""

    def __init__(self, terminal: dict[str, bool | None], attempts: dict[str, int | None]) -> None:
        self._terminal = terminal
        self._attempts = attempts

    def terminal_record_present(self, *, tenant_id: str, task_id: str) -> bool | None:
        del tenant_id
        return self._terminal.get(task_id)

    def attempts_observed(self, *, tenant_id: str, task_id: str) -> int | None:
        del tenant_id
        return self._attempts.get(task_id)


def _uncertain(schedule_id: str, task_id: str) -> ScheduledResume:
    return ScheduledResume(
        schedule_id=schedule_id,
        task_id=task_id,
        tenant_id="t-rec06",
        resume_token=f"tok-{task_id}",
        run_at_utc=T0.isoformat(),
        status=ScheduledResumeStatus.UNCERTAIN,
        fence=3,
    )


def test_rw_rec_06_uncertain_entries_reconciled_exactly_once() -> None:
    from intergrax.runtime.long_running.uncertain_reconciler import (
        DoubleExecutionCheck,
        UncertainOutcome,
        UncertainReconciler,
    )

    source = _InMemoryUncertainSource(
        [
            _uncertain("s-terminal", "task-done"),  # terminal record present
            _uncertain("s-nothing", "task-never"),  # nothing recorded at all
            _uncertain("s-partial", "task-partial"),  # attempt started, no terminal record
        ]
    )
    evidence = _Evidence(
        terminal={"task-done": True, "task-never": False, "task-partial": False},
        attempts={"task-done": 1, "task-never": 0, "task-partial": 1},
    )
    reconciler = UncertainReconciler(source, evidence)

    first = reconciler.reconcile_once(T0)

    by_id = {record.schedule_id: record for record in first}
    assert sorted(by_id) == ["s-nothing", "s-partial", "s-terminal"]
    assert by_id["s-terminal"].outcome is UncertainOutcome.EXECUTION_PROVEN
    assert by_id["s-nothing"].outcome is UncertainOutcome.NOT_STARTED_PROVEN
    assert by_id["s-partial"].outcome is UncertainOutcome.UNPROVEN
    assert [by_id[s].resolve_applied for s in ("s-terminal", "s-nothing", "s-partial")] == [True, True, False]
    assert [by_id[s].redrive_eligible for s in ("s-terminal", "s-nothing", "s-partial")] == [False, True, False]
    assert {s: by_id[s].attempts_observed for s in by_id} == {"s-terminal": 1, "s-nothing": 0, "s-partial": 1}
    assert all(record.double_execution is DoubleExecutionCheck.AT_MOST_ONCE for record in first)
    # Only the proven not-started entry goes back to PENDING (one redrive); the unproven one stays.
    assert source.rows["s-terminal"].status is ScheduledResumeStatus.COMPLETED
    assert source.rows["s-nothing"].status is ScheduledResumeStatus.PENDING
    assert source.rows["s-partial"].status is ScheduledResumeStatus.UNCERTAIN
    assert sorted(source.resolve_calls) == [("s-nothing", "not_started_proven"), ("s-terminal", "execution_proven")]

    second = reconciler.reconcile_once(T0 + timedelta(seconds=30))

    assert [record.schedule_id for record in second] == ["s-partial"], "resolved entries were reconciled twice"
    assert second[0].resolve_applied is False
    assert len(source.resolve_calls) == 2


def test_rw_rec_06b_double_execution_is_recorded_and_never_redriven() -> None:
    from intergrax.runtime.long_running.uncertain_reconciler import (
        DoubleExecutionCheck,
        UncertainOutcome,
        UncertainReconciler,
    )

    source = _InMemoryUncertainSource([_uncertain("s-twice", "task-twice"), _uncertain("s-blind", "task-blind")])
    evidence = _Evidence(terminal={"task-twice": True, "task-blind": None}, attempts={"task-twice": 2})

    records = {r.schedule_id: r for r in UncertainReconciler(source, evidence).reconcile_once(T0)}

    assert records["s-twice"].double_execution is DoubleExecutionCheck.VIOLATED
    assert records["s-twice"].attempts_observed == 2
    assert records["s-blind"].double_execution is DoubleExecutionCheck.NOT_OBSERVABLE
    assert {r.outcome for r in records.values()} == {UncertainOutcome.UNPROVEN}
    assert source.resolve_calls == []
    assert {row.status for row in source.rows.values()} == {ScheduledResumeStatus.UNCERTAIN}


async def test_rw_rec_06c_scheduler_runs_the_reconciler_as_an_isolated_phase(tmp_path: Path) -> None:
    from intergrax.runtime.long_running.uncertain_reconciler import UncertainReconciler

    source = _InMemoryUncertainSource([_uncertain("s-nothing", "task-never")])
    reconciler = UncertainReconciler(source, _Evidence(terminal={"task-never": False}, attempts={"task-never": 0}))
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "rec06c.db")
    scheduler = LongRunningScheduler(store, RecordingExecutor(), schedule_store=store, uncertain_reconciler=reconciler)

    assert await scheduler.tick(now=T0) == 0
    assert source.rows["s-nothing"].status is ScheduledResumeStatus.PENDING


# ---------------------------------------------------------------- REC-07


class _RecordingStore(SQLiteTaskCheckpointStore):
    """SQLite schedule store that counts releases and skips and can fail the eligibility read."""

    def __init__(self, *, db_path: Path) -> None:
        super().__init__(db_path=db_path)
        self.releases = 0
        self.skips = 0
        self.fail_eligibility_read = False

    def release_claim(self, claim: ScheduledResumeClaim) -> None:
        self.releases += 1
        super().release_claim(claim)

    def skip_claim(self, claim: ScheduledResumeClaim) -> None:
        self.skips += 1
        super().skip_claim(claim)

    def get_by_token(self, task_id: str, tenant_id: str, resume_token: str) -> TaskCheckpoint | None:
        if self.fail_eligibility_read:
            raise RuntimeError("permanent checkpoint read error")
        return super().get_by_token(task_id, tenant_id, resume_token)


class _FailingMaterializer(RecurrenceMaterializer):
    def __init__(self, schedule_store: SQLiteTaskCheckpointStore, error: Exception) -> None:
        super().__init__(schedule_store)
        self.error = error
        self.calls = 0

    def materialize_next(self, entry: ScheduledResume, *, now: datetime) -> list[ScheduledResume]:
        del entry, now
        self.calls += 1
        raise self.error


def _start_chain(store: SQLiteTaskCheckpointStore) -> None:
    checkpoint = save_paused(store)
    rule = RecurrenceRule(
        rule_id="rule-rw-rec-07",
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        resume_token=checkpoint.resume_token,
        interval_seconds=60,
        start_utc=T0.isoformat(),
    )
    assert RecurrenceMaterializer(store).start(rule) is not None


async def _tick_many(scheduler: LongRunningScheduler, ticks: int) -> None:
    for second in range(1, ticks + 1):
        await scheduler.tick(now=T0 + timedelta(seconds=second))


def _cap_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR and _CAP_CODE in r.getMessage()]


@pytest.mark.parametrize("failure", ["materialization", "eligibility"])
async def test_rw_rec_07_permanent_failure_release_is_capped(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    store = _RecordingStore(db_path=tmp_path / f"rec07-{failure}.db")
    _start_chain(store)
    materializer = _FailingMaterializer(store, RuntimeError("permanent store error"))
    executor = RecordingExecutor()
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        owner_id="sched-rw-rec-07",
        recurrence=materializer,
        max_claim_release_attempts=3,
    )
    store.fail_eligibility_read = failure == "eligibility"

    with caplog.at_level(logging.ERROR):
        await _tick_many(scheduler, 8)

    assert store.releases == 3, "the claim was released without a cap"
    assert store.skips == 1
    assert executor.fired == []
    assert len(_cap_logs(caplog)) == 1


async def test_rw_rec_07c_overflow_is_mapped_to_skip(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = _RecordingStore(db_path=tmp_path / "rec07c.db")
    _start_chain(store)
    materializer = _FailingMaterializer(store, OverflowError("date out of range"))
    executor = RecordingExecutor()
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        owner_id="sched-rw-rec-07c",
        recurrence=materializer,
    )

    with caplog.at_level(logging.ERROR):
        await _tick_many(scheduler, 3)

    assert (store.releases, store.skips, materializer.calls) == (0, 1, 1)
    assert executor.fired == []
    assert _cap_logs(caplog) == []
