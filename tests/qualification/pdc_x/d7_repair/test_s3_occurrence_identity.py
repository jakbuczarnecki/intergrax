# © Artur Czarnecki. All rights reserved.

"""F-D7-04 — the occurrence identity is tenant- and task-qualified; conflicts are typed."""

from __future__ import annotations

from datetime import timedelta

import pytest
from intergrax.runtime.long_running import recurrence as rec
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer, RecurrenceRule
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from tests.qualification.pdc_x.d7_repair._support import FAR_FUTURE, RESUME_TOKEN, T0

pytestmark = [pytest.mark.unit]


def _rule(tenant: str, task: str = "task-1", rule_id: str = "daily-report", **overrides) -> RecurrenceRule:
    params = {
        "rule_id": rule_id,
        "tenant_id": tenant,
        "task_id": task,
        "resume_token": RESUME_TOKEN,
        "interval_seconds": 3600,
        "start_utc": T0.isoformat(),
    }
    params.update(overrides)
    return RecurrenceRule(**params)


def test_two_tenants_with_the_same_rule_id_and_start_both_get_their_chain(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "s3.db")
    materializer = RecurrenceMaterializer(store)
    first = materializer.start(_rule("tenant-a"))
    second = materializer.start(_rule("tenant-b"))
    assert first is not None
    assert second is not None  # was silently dropped (returned None) before the repair
    assert first.schedule_id != second.schedule_id
    pending = store.list_due(before_utc_iso=FAR_FUTURE)
    assert sorted(row.tenant_id for row in pending) == ["tenant-a", "tenant-b"]


def test_same_rule_started_twice_reports_already_exists(tmp_path) -> None:
    materializer = RecurrenceMaterializer(SQLiteTaskCheckpointStore(db_path=tmp_path / "s3b.db"))
    rule = _rule("tenant-a")
    assert materializer.start_checked(rule).value == "started"
    assert materializer.start_checked(rule).value == "already_exists"
    assert materializer.start(rule) is None  # the legacy return type is unchanged


def test_exhausted_rule_reports_no_occurrence_not_a_conflict(tmp_path) -> None:
    materializer = RecurrenceMaterializer(SQLiteTaskCheckpointStore(db_path=tmp_path / "s3c.db"))
    # The only cron instant after the start (2026-03-02 00:00 UTC) lies beyond ``end_utc``.
    ended = _rule(
        "tenant-a",
        rule_id="ended",
        interval_seconds=None,
        cron="0 0 * * *",
        end_utc=(T0 + timedelta(hours=1)).isoformat(),
    )
    assert materializer.start_checked(ended).value == "no_occurrence"


def test_occurrence_id_is_deterministic_qualified_and_leaks_neither_tenant_nor_task() -> None:
    a = rec.occurrence_schedule_id("daily-report", T0, tenant_id="tenant-a", task_id="task-1")
    assert a == rec.occurrence_schedule_id("daily-report", T0, tenant_id="tenant-a", task_id="task-1")
    assert a != rec.occurrence_schedule_id("daily-report", T0, tenant_id="tenant-b", task_id="task-1")
    assert a != rec.occurrence_schedule_id("daily-report", T0, tenant_id="tenant-a", task_id="task-2")
    later = T0 + timedelta(seconds=1)
    assert a != rec.occurrence_schedule_id("daily-report", later, tenant_id="tenant-a", task_id="task-1")
    assert a.startswith("rec_") and a.endswith("_daily-report_20260301T100000Z")
    assert "tenant-a" not in a and "task-1" not in a


def test_occurrence_id_requires_tenant_and_task() -> None:
    with pytest.raises(TypeError):
        rec.occurrence_schedule_id("daily-report", T0)  # type: ignore[call-arg]


def test_occurrence_id_literal_is_pinned() -> None:
    """The id format is persisted state: changing it silently would orphan live chains."""
    got = rec.occurrence_schedule_id("daily-report", T0, tenant_id="tenant-a", task_id="task-1")
    assert got == "rec_bea5214118012898_96b08478ce8f3ce5_daily-report_20260301T100000Z"
