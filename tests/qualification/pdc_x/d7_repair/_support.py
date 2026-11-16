# © Artur Czarnecki. All rights reserved.

"""Shared helpers for the T-IGX-041-G (PDC-X P7, D7 repair) pinned regression tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskResult
from tests.qualification.pdc_x.d7._support import (
    RESUME_TOKEN,
    ok_task_result,
    paused_checkpoint,
)

T0 = datetime(2026, 3, 1, 10, 0, 0, tzinfo=UTC)
FAR_FUTURE = "9999-12-31T00:00:00+00:00"


def ledger_status(db: Path) -> dict[str, str]:
    """``schedule_id -> status`` straight from the SQLite ledger (read-only probe)."""
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT schedule_id, status FROM scheduled_resumes").fetchall()
    return {schedule_id: status for schedule_id, status in rows}


def save_paused(store: SQLiteTaskCheckpointStore, *, tenant_id: str = "t-d7") -> TaskCheckpoint:
    return store.save(paused_checkpoint(tenant_id=tenant_id))


def schedule_one_shot(
    store: SQLiteTaskCheckpointStore,
    checkpoint: TaskCheckpoint,
    *,
    run_at: datetime,
    schedule_id: str | None = None,
) -> ScheduledResume:
    params: dict[str, Any] = {
        "task_id": checkpoint.task_id,
        "tenant_id": checkpoint.tenant_id,
        "resume_token": checkpoint.resume_token,
        "run_at_utc": run_at.isoformat(),
    }
    if schedule_id is not None:
        params["schedule_id"] = schedule_id
    return store.schedule(ScheduledResume(**params))


class RecordingExecutor:
    """Resume executor that records fires and can be told to fail for chosen schedules."""

    def __init__(self, *, fail_for: frozenset[str] = frozenset()) -> None:
        self.fired: list[str] = []
        self.fail_for = fail_for
        self.on_fire: Callable[[str, TaskCheckpoint], None] | None = None

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        schedule_id = str(task.metadata.get("schedule_id"))
        if schedule_id in self.fail_for:
            raise RuntimeError(f"resume failed for {schedule_id}")
        if self.on_fire is not None:
            self.on_fire(schedule_id, checkpoint)
        self.fired.append(schedule_id)
        return ok_task_result(checkpoint.task_id)


__all__ = [
    "FAR_FUTURE",
    "RESUME_TOKEN",
    "T0",
    "RecordingExecutor",
    "ledger_status",
    "save_paused",
    "schedule_one_shot",
    "timedelta",
]
