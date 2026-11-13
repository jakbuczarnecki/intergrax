# © Artur Czarnecki. All rights reserved.

"""PDC-D7-R006 / R007 / R008 — recurrence on the existing SchedulerLedger.

Real ``SQLiteTaskCheckpointStore`` + real ``LongRunningScheduler``; no mocks of the
unit under test. Time is a frozen ``now`` passed to ``tick``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from intergrax.runtime.long_running.recurrence import (
    MAX_MISFIRE_SCAN,
    MisfirePolicy,
    RecurrenceMaterializer,
    RecurrenceRule,
    occurrence_schedule_id,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7._support import (
    RESUME_TOKEN,
    TENANT_ID,
    CountingResumeExecutor,
    build_scheduler,
    drain_ticks,
    paused_checkpoint,
)

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FAR_FUTURE = "9999-12-31T00:00:00+00:00"
_WARSAW = ZoneInfo("Europe/Warsaw")
T0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _rule(**overrides) -> RecurrenceRule:
    params = {
        "rule_id": "rule-d7",
        "tenant_id": TENANT_ID,
        "task_id": "task-d7",
        "resume_token": RESUME_TOKEN,
        "start_utc": T0.isoformat(),
    }
    params.update(overrides)
    return RecurrenceRule(**params)


def _pending_ids(store: SQLiteTaskCheckpointStore) -> list[str]:
    return [row.schedule_id for row in store.list_due(before_utc_iso=_FAR_FUTURE)]


# --------------------------------------------------------------------------- R006


def test_interval_rule_enumerates_occurrences_in_utc() -> None:
    rule = _rule(interval_seconds=900)
    assert rule.first_occurrence() == T0
    assert rule.next_after(T0) == T0 + timedelta(minutes=15)
    assert rule.next_after(T0 + timedelta(seconds=1)) == T0 + timedelta(minutes=15)
    assert rule.next_after(T0 - timedelta(hours=1)) == T0
    got = rule.occurrences_between(T0 - timedelta(seconds=1), T0 + timedelta(hours=1), 100)
    assert got == [T0 + timedelta(minutes=15 * k) for k in range(5)]


def test_cron_rule_enumerates_weekday_business_hours_in_zone() -> None:
    rule = _rule(
        cron="*/15 9-10 * * 1-5",
        timezone="Europe/Warsaw",
        start_utc=_utc(2026, 1, 5).isoformat(),  # Monday, CET (UTC+1)
    )
    first_day = rule.occurrences_between(_utc(2026, 1, 4), _utc(2026, 1, 5, 23), 100)
    assert first_day == [
        _utc(2026, 1, 5, 8, 0),
        _utc(2026, 1, 5, 8, 15),
        _utc(2026, 1, 5, 8, 30),
        _utc(2026, 1, 5, 8, 45),
        _utc(2026, 1, 5, 9, 0),
        _utc(2026, 1, 5, 9, 15),
        _utc(2026, 1, 5, 9, 30),
        _utc(2026, 1, 5, 9, 45),
    ]
    # Friday's last slot rolls over the weekend to Monday morning.
    assert rule.next_after(_utc(2026, 1, 9, 9, 45)) == _utc(2026, 1, 12, 8, 0)


def test_cron_day_of_month_and_day_of_week_use_or_semantics() -> None:
    rule = _rule(cron="0 12 13 * 5", timezone="UTC", start_utc=_utc(2026, 1, 1).isoformat())
    # 2026-01-02 is a Friday: matches through day-of-week even though dom is 13.
    assert rule.first_occurrence() == _utc(2026, 1, 2, 12, 0)
    assert rule.next_after(_utc(2026, 1, 9, 12, 0)) == _utc(2026, 1, 13, 12, 0)


def test_cron_leap_day_rule_finds_next_leap_year() -> None:
    rule = _rule(cron="0 0 29 2 *", timezone="UTC", start_utc=_utc(2026, 1, 1).isoformat())
    assert rule.first_occurrence() == _utc(2028, 2, 29)


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # neither interval nor cron
        {"interval_seconds": 60, "cron": "* * * * *"},  # both
        {"interval_seconds": 0},
        {"cron": "61 * * * *"},
        {"cron": "* * * *"},
        {"cron": "*/0 * * * *"},
        {"cron": "* * * * *", "timezone": "Mars/Olympus_Mons"},
        {"interval_seconds": 60, "start_utc": "2026-03-01T10:00:00"},  # naive timestamp
        {"interval_seconds": 60, "end_utc": "2026-02-01T00:00:00+00:00"},  # end before start
        {"interval_seconds": 60, "resume_metadata": {"approver": "x"}},  # authority-bearing
    ],
)
def test_invalid_rules_are_rejected_fail_closed(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _rule(**overrides)


def test_misfire_policy_shapes_are_validated() -> None:
    assert MisfirePolicy.skip().max_catch_up == 0
    assert MisfirePolicy.catch_up(3).max_catch_up == 3
    with pytest.raises(ValueError):
        MisfirePolicy.catch_up(0)


async def test_scheduler_materializes_next_one_shot_after_each_occurrence(tmp_path) -> None:
    executor = CountingResumeExecutor()
    scheduler, store, materializer = build_scheduler(tmp_path / "rec.db", executor)
    store.save(paused_checkpoint())
    # The recurrence targets the paused checkpoint's own task id.
    rule = _rule(interval_seconds=60, task_id=store.list_paused()[0].task_id)

    first = materializer.start(rule)
    assert first is not None
    assert first.schedule_id == occurrence_schedule_id(rule.rule_id, T0)
    assert _pending_ids(store) == [first.schedule_id]  # exactly one materialized occurrence

    expected_ids: list[str] = []
    for k in range(3):
        occurrence = T0 + timedelta(seconds=60 * k)
        expected_ids.append(occurrence_schedule_id(rule.rule_id, occurrence))
        fired = await drain_ticks(scheduler, now=occurrence + timedelta(seconds=1))
        assert fired == 1
        # After each occurrence exactly the NEXT one-shot is materialized (no pile-up).
        assert _pending_ids(store) == [occurrence_schedule_id(rule.rule_id, occurrence + timedelta(seconds=60))]

    assert executor.schedule_ids == expected_ids
    assert executor.occurrences == [(T0 + timedelta(seconds=60 * k)).isoformat() for k in range(3)]


def test_each_occurrence_is_a_plain_scheduled_resume_with_utc_storage(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "utc.db")
    materializer = RecurrenceMaterializer(store)
    rule = _rule(cron="0 9 * * *", timezone="Europe/Warsaw")
    entry = materializer.start(rule)
    assert entry is not None
    assert entry.run_at_utc.endswith("+00:00")
    assert datetime.fromisoformat(entry.run_at_utc).utcoffset() == timedelta(0)
    assert entry.tenant_id == TENANT_ID and entry.resume_token == RESUME_TOKEN
    assert entry.resume_metadata["recurrence_occurrence_utc"] == entry.run_at_utc
    # 09:00 Warsaw on 2026-03-02 (CET) is 08:00 UTC.
    assert entry.run_at_utc == _utc(2026, 3, 2, 8, 0).isoformat()


def test_materialization_is_idempotent_by_deterministic_occurrence_id(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "idem.db")
    materializer = RecurrenceMaterializer(store)
    rule = _rule(interval_seconds=60)
    entry = materializer.start(rule)
    assert entry is not None
    now = T0 + timedelta(seconds=1)
    created = materializer.materialize_next(entry, now=now)
    assert [row.run_at_utc for row in created] == [(T0 + timedelta(seconds=60)).isoformat()]
    assert materializer.materialize_next(entry, now=now) == []  # duplicate occurrence rejected
    assert materializer.start(rule) is None  # first occurrence already materialized
    assert len(_pending_ids(store)) == 2


def test_end_utc_terminates_the_chain(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "end.db")
    materializer = RecurrenceMaterializer(store)
    rule = _rule(interval_seconds=60, end_utc=(T0 + timedelta(seconds=90)).isoformat())
    entry0 = materializer.start(rule)
    assert entry0 is not None
    (entry1,) = materializer.materialize_next(entry0, now=T0 + timedelta(seconds=1))
    assert entry1.run_at_utc == (T0 + timedelta(seconds=60)).isoformat()
    assert materializer.materialize_next(entry1, now=T0 + timedelta(seconds=61)) == []


def test_recurrence_adds_no_second_scheduler_or_engine() -> None:
    recurrence_src = (_REPO_ROOT / "intergrax" / "runtime" / "long_running" / "recurrence.py").read_text(
        encoding="utf-8"
    )
    for token in (
        "asyncio",
        "threading",
        "create_task",
        "sleep(",
        "while True",
        "getattr(",
        "hasattr(",
        "setattr(",
        "croniter",
        "apscheduler",
        "celery",
        "from agents",
        "from applications",
        "ExecutionRuntime(",
        "NexusLoop(",
    ):
        assert token not in recurrence_src, token
    scheduler_src = (_REPO_ROOT / "intergrax" / "runtime" / "long_running" / "scheduler.py").read_text(encoding="utf-8")
    assert scheduler_src.count("def _run_loop") == 1  # the one existing poll loop


# --------------------------------------------------------------------------- R007 (DST)


def test_spring_forward_gap_fires_once_at_first_valid_instant() -> None:
    # Europe/Warsaw: 2027-03-28 02:00 CET jumps to 03:00 CEST; 02:30 does not exist.
    rule = _rule(cron="30 2 * * *", timezone="Europe/Warsaw", start_utc=_utc(2027, 3, 26).isoformat())
    got = rule.occurrences_between(_utc(2027, 3, 25), _utc(2027, 3, 30), 100)
    assert got == [
        _utc(2027, 3, 26, 1, 30),
        _utc(2027, 3, 27, 1, 30),
        _utc(2027, 3, 28, 1, 30),  # 03:30 CEST: shifted by the gap, not skipped
        _utc(2027, 3, 29, 0, 30),
    ]
    local = [(when.astimezone(_WARSAW).hour, when.astimezone(_WARSAW).minute) for when in got]
    assert local == [(2, 30), (2, 30), (3, 30), (2, 30)]


def test_fall_back_overlap_fires_exactly_once() -> None:
    # Europe/Warsaw: 2026-10-25 03:00 CEST falls back to 02:00 CET; 02:30 occurs twice.
    rule = _rule(cron="30 2 * * *", timezone="Europe/Warsaw", start_utc=_utc(2026, 10, 23).isoformat())
    got = rule.occurrences_between(_utc(2026, 10, 22), _utc(2026, 10, 27), 100)
    assert got == [
        _utc(2026, 10, 23, 0, 30),
        _utc(2026, 10, 24, 0, 30),
        _utc(2026, 10, 25, 0, 30),  # first 02:30 only (CEST); the CET repeat (01:30Z) is not fired
        _utc(2026, 10, 26, 1, 30),
    ]


def test_wildcard_hourly_rule_follows_elapsed_time_across_transitions() -> None:
    fall_rule = _rule(cron="30 * * * *", timezone="Europe/Warsaw", start_utc=_utc(2026, 10, 20).isoformat())
    fall = fall_rule.occurrences_between(_utc(2026, 10, 24, 22, 0), _utc(2026, 10, 25, 23, 0), 100)
    assert len(fall) == 25  # the repeated hour fires twice: 25 hours in the local day
    assert all(b - a == timedelta(hours=1) for a, b in zip(fall, fall[1:]))

    spring_rule = _rule(cron="30 * * * *", timezone="Europe/Warsaw", start_utc=_utc(2027, 3, 20).isoformat())
    spring = spring_rule.occurrences_between(_utc(2027, 3, 27, 23, 0), _utc(2027, 3, 28, 22, 0), 100)
    assert len(spring) == 23  # the missing hour does not fire (no duplicate shifted slot)
    assert all(b - a == timedelta(hours=1) for a, b in zip(spring, spring[1:]))


def test_interval_rule_is_elapsed_time_and_dst_immune() -> None:
    rule = _rule(
        interval_seconds=86400,
        timezone="Europe/Warsaw",
        start_utc=_utc(2027, 3, 26, 12).isoformat(),
    )
    got = rule.occurrences_between(_utc(2027, 3, 26), _utc(2027, 3, 31), 100)
    assert len(got) == 5
    assert all(b - a == timedelta(days=1) for a, b in zip(got, got[1:]))
    hours = [when.astimezone(_WARSAW).hour for when in got]
    assert hours[0] == 13 and hours[-1] == 14  # local wall clock shifts; elapsed time does not


# --------------------------------------------------------------------------- R008 (misfire)


@pytest.mark.parametrize(
    ("policy", "expected_offsets"),
    [
        (MisfirePolicy.skip(), [360]),
        (MisfirePolicy.fire_once(), [300]),
        (MisfirePolicy.catch_up(3), [180, 240, 300]),
    ],
)
def test_misfire_policy_decides_which_missed_occurrences_are_materialized(
    tmp_path, policy: MisfirePolicy, expected_offsets: list[int]
) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "misfire.db")
    materializer = RecurrenceMaterializer(store)
    rule = _rule(interval_seconds=60, misfire=policy)
    entry0 = materializer.start(rule)
    assert entry0 is not None
    # The scheduler was down: occurrences +60..+300 were missed, now is +330.
    created = materializer.materialize_next(entry0, now=T0 + timedelta(seconds=330))
    assert [row.run_at_utc for row in created] == [
        (T0 + timedelta(seconds=offset)).isoformat() for offset in expected_offsets
    ]


@pytest.mark.parametrize(
    ("policy", "expected_resumes"),
    [
        (MisfirePolicy.skip(), 1),
        (MisfirePolicy.fire_once(), 2),
        (MisfirePolicy.catch_up(3), 4),
        (MisfirePolicy.catch_up(10), 6),
    ],
)
async def test_scheduler_applies_misfire_policy_end_to_end(
    tmp_path, policy: MisfirePolicy, expected_resumes: int
) -> None:
    executor = CountingResumeExecutor()
    scheduler, store, materializer = build_scheduler(tmp_path / "e2e.db", executor)
    store.save(paused_checkpoint())
    rule = _rule(
        interval_seconds=60,
        misfire=policy,
        task_id=store.list_paused()[0].task_id,
    )
    materializer.start(rule)
    now = T0 + timedelta(seconds=330)
    fired = await drain_ticks(scheduler, now=now)
    assert fired == expected_resumes
    assert len(set(executor.schedule_ids)) == expected_resumes  # no occurrence fired twice
    # Whatever the policy, the chain continues with the first future occurrence.
    assert _pending_ids(store) == [occurrence_schedule_id(rule.rule_id, T0 + timedelta(seconds=360))]


def test_misfire_scan_overflow_degrades_to_skip(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "overflow.db")
    materializer = RecurrenceMaterializer(store)
    rule = _rule(interval_seconds=1, misfire=MisfirePolicy.fire_once())
    entry0 = materializer.start(rule)
    assert entry0 is not None
    now = T0 + timedelta(seconds=MAX_MISFIRE_SCAN + 5000)
    created = materializer.materialize_next(entry0, now=now)
    assert [row.run_at_utc for row in created] == [(now + timedelta(seconds=1)).isoformat()]
