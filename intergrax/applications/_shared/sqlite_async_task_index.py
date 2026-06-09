# © Artur Czarnecki. All rights reserved.

"""SQLite-backed durable async task index scaffold (IDEAL-3.4 / IDEAL-28.2)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from intergrax.applications._shared.async_task_dispatch import AsyncTaskHandle
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


class SqliteAsyncTaskIndex:
    """Process-local durable index; suitable for single-host product scaffolds."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, asyncio.Task[TaskResult]] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS async_task_handles (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _persist(self, handle: AsyncTaskHandle) -> None:
        payload = {
            "task_id": handle.task_id,
            "status": handle.status,
            "error": handle.error,
            "state": handle.result.state.value if handle.result else None,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO async_task_handles(task_id, status, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json
                """,
                (handle.task_id, handle.status, json.dumps(payload)),
            )

    def get(self, task_id: str) -> AsyncTaskHandle | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, payload_json FROM async_task_handles WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        status, payload_json = row
        payload = json.loads(payload_json)
        return AsyncTaskHandle(
            task_id=task_id,
            status=status,
            error=payload.get("error"),
        )

    async def enqueue(self, runner: UnifiedTaskRunner, task: Task) -> AsyncTaskHandle:
        task_id = task.task_id
        handle = AsyncTaskHandle(task_id=task_id, status="pending")
        self._persist(handle)

        async def _run() -> TaskResult:
            running = AsyncTaskHandle(task_id=task_id, status="running")
            self._persist(running)
            try:
                result = await runner.run_task(task)
            except Exception as exc:
                failed = AsyncTaskHandle(
                    task_id=task_id,
                    status="failed",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
                self._persist(failed)
                raise
            completed = AsyncTaskHandle(
                task_id=task_id,
                status=result.state.value,
                result=result,
            )
            self._persist(completed)
            return result

        self._tasks[task_id] = asyncio.create_task(_run())
        return handle
