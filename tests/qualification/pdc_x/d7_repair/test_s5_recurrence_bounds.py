# © Artur Czarnecki. All rights reserved.

"""F-D7-11 (+ by-products F-D7-18, F-D7-21) — bounded recurrence, no minting via plain metadata."""

from __future__ import annotations

from datetime import timedelta

import pytest
from intergrax.runtime.long_running import recurrence as rec
from intergrax.runtime.long_running.recurrence import (
    RECURRENCE_OCCURRENCE_METADATA_KEY,
    RECURRENCE_RULE_METADATA_KEY,
    MisfirePolicy,
    RecurrenceMaterializer,
    RecurrenceRule,
)
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.scheduled_resume_metadata import ScheduledResumeMetadataValidationError
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import (
    FAR_FUTURE,
    RESUME_TOKEN,
    T0,
    RecordingExecutor,
    save_paused,
)

pytestmark = [pytest.mark.unit]


def _rule(checkpoint=None, **overrides) -> RecurrenceRule:
    params = {
        "rule_id": "rule-s5",
        "tenant_id": checkpoint.tenant_id if checkpoint else "t-d7",
        "task_id": checkpoint.task_id if checkpoint else "task-s5",
        "resume_token": RESUME_TOKEN,
        "interval_seconds": 60,
        "start_utc": T0.isoformat(),
    }
    params.update(overrides)
    return RecurrenceRule(**params)


# ------------------------------------------------------------------ smuggling


def test_schedule_resume_rejects_recurrence_keys_in_plain_metadata(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s5.db")
    scheduler = LongRunningScheduler(store, RecordingExecutor(), schedule_store=store, owner_id="s5")  # type: ignore[arg-type]
    rule_dump = _rule().model_dump(mode="json")
    for key in (RECURRENCE_RULE_METADATA_KEY, RECURRENCE_OCCURRENCE_METADATA_KEY):
        with pytest.raises(ScheduledResumeMetadataValidationError):
            scheduler.schedule_resume(
                task_id="task-s5",
                tenant_id="t-d7",
                resume_token=RESUME_TOKEN,
                run_at_utc=T0.isoformat(),
                resume_metadata={key: rule_dump if key == RECURRENCE_RULE_METADATA_KEY else T0.isoformat()},
            )
    assert store.list_due(before_utc_iso=FAR_FUTURE) == []


def test_rule_resume_metadata_cannot_carry_recurrence_keys() -> None:
    with pytest.raises(ValueError):
        _rule(resume_metadata={RECURRENCE_OCCURRENCE_METADATA_KEY: T0.isoformat()})


def _hand_built(rule: RecurrenceRule, **overrides) -> ScheduledResume:
    """A row that carries recurrence metadata but is not what the materializer would write."""
    when = T0.isoformat()
    params = {
        "schedule_id": rec.occurrence_schedule_id(rule.rule_id, T0, tenant_id=rule.tenant_id, task_id=rule.task_id),
        "task_id": rule.task_id,
        "tenant_id": rule.tenant_id,
        "resume_token": rule.resume_token,
        "run_at_utc": when,
        "resume_metadata": {
            RECURRENCE_RULE_METADATA_KEY: rule.model_dump(mode="json"),
            RECURRENCE_OCCURRENCE_METADATA_KEY: when,
        },
    }
    params.update(overrides)
    return ScheduledResume(**params)


def test_persistence_accepts_only_self_consistent_occurrence_rows(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s5b.db")
    rule = _rule()
    consistent = _hand_built(rule)
    assert store.schedule(consistent).schedule_id == consistent.schedule_id  # same shape the materializer writes

    other = _rule(rule_id="rule-other")
    forged = {
        "arbitrary id": _hand_built(rule, schedule_id="sched-arbitrary"),
        "wrong tenant": _hand_built(rule, tenant_id="someone-else"),
        "wrong task": _hand_built(rule, task_id="another-task"),
        "wrong run_at": _hand_built(rule, run_at_utc=(T0 + timedelta(seconds=7)).isoformat()),
        "not an occurrence of the rule": _hand_built(
            rule,
            run_at_utc=(T0 + timedelta(seconds=7)).isoformat(),
            schedule_id=rec.occurrence_schedule_id(
                rule.rule_id, T0 + timedelta(seconds=7), tenant_id=rule.tenant_id, task_id=rule.task_id
            ),
            resume_metadata={
                RECURRENCE_RULE_METADATA_KEY: rule.model_dump(mode="json"),
                RECURRENCE_OCCURRENCE_METADATA_KEY: (T0 + timedelta(seconds=7)).isoformat(),
            },
        ),
        "id of another rule": _hand_built(
            rule,
            schedule_id=rec.occurrence_schedule_id(other.rule_id, T0, tenant_id=rule.tenant_id, task_id=rule.task_id),
        ),
        "occurrence key only": _hand_built(rule, resume_metadata={RECURRENCE_OCCURRENCE_METADATA_KEY: T0.isoformat()}),
        "rule key only": _hand_built(
            rule, resume_metadata={RECURRENCE_RULE_METADATA_KEY: rule.model_dump(mode="json")}
        ),
    }
    for label, entry in forged.items():
        with pytest.raises(ValueError):
            store.schedule(entry)
        assert label  # the failing case is named in the parametrized-by-loop report


def test_materializer_rows_still_persist_through_the_chain(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s5c.db")
    materializer = RecurrenceMaterializer(store)
    first = materializer.start(_rule(cron="*/5 * * * *", interval_seconds=None, timezone="Europe/Warsaw"))
    assert first is not None
    (second,) = materializer.materialize_next(first, now=T0)
    assert second.run_at_utc > first.run_at_utc


# ------------------------------------------------------------------ bounds


def test_interval_and_catch_up_bounds_are_enforced() -> None:
    assert rec.MIN_INTERVAL_SECONDS >= 1
    _rule(interval_seconds=rec.MIN_INTERVAL_SECONDS)
    _rule(interval_seconds=rec.MAX_INTERVAL_SECONDS)
    with pytest.raises(ValueError):
        _rule(interval_seconds=rec.MIN_INTERVAL_SECONDS - 1)
    with pytest.raises(ValueError):
        _rule(interval_seconds=rec.MAX_INTERVAL_SECONDS + 1)
    with pytest.raises(ValueError):
        _rule(interval_seconds=10**15)  # raised OverflowError at evaluation before the repair
    assert MisfirePolicy.catch_up(rec.MAX_CATCH_UP).max_catch_up == rec.MAX_CATCH_UP
    with pytest.raises(ValueError):
        MisfirePolicy.catch_up(rec.MAX_CATCH_UP + 1)


def test_unsatisfiable_cron_is_rejected_at_validation() -> None:
    with pytest.raises(ValueError):
        _rule(interval_seconds=None, cron="0 0 30 2 *")
    _rule(interval_seconds=None, cron="0 0 29 2 *")  # leap day is satisfiable


async def test_catch_up_drain_work_is_bounded(tmp_path) -> None:
    class _CountingStore(SQLiteTaskCheckpointStore):
        schedule_calls = 0

        def schedule(self, entry):
            type(self).schedule_calls += 1
            return super().schedule(entry)

    store = _CountingStore(db_path=tmp_path / "drain.db")
    checkpoint = save_paused(store)
    rule = _rule(checkpoint, interval_seconds=1, misfire=MisfirePolicy.catch_up(rec.MAX_CATCH_UP))
    RecurrenceMaterializer(store).start(rule)
    executor = RecordingExecutor()
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,  # type: ignore[arg-type]
        owner_id="s5",
        recurrence=RecurrenceMaterializer(store),  # type: ignore[arg-type]
    )
    now = T0 + timedelta(seconds=3600)
    for _ in range(rec.MAX_CATCH_UP + 10):
        if await scheduler.tick(now=now) == 0:
            break
    assert len(executor.fired) <= rec.MAX_CATCH_UP + 2
    assert _CountingStore.schedule_calls <= (rec.MAX_CATCH_UP + 2) ** 2  # measured by calls, not wall clock


def test_dst_gap_wording_matches_the_behaviour() -> None:
    doc = rec.__doc__ or ""
    assert "first valid instant after the gap" not in doc
    assert "gap length" in doc
