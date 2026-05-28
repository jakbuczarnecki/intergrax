# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite implementation of RuntimeEventPersistence (Tier-0 default backend)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent


class SQLiteRuntimeEventStore(RuntimeEventPersistence):
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_events_run
                ON runtime_events (tenant_id, run_id, id ASC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_events_task
                ON runtime_events (tenant_id, task_id, id ASC);
                """
            )

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        scope = tenant_id or event.tenant_id or ""
        payload = event.model_dump(mode="json")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO runtime_events (
                    tenant_id, task_id, run_id, event_id, event_type,
                    created_at_utc, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    event.task_id,
                    event.run_id,
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    json.dumps(payload),
                ),
            )

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT event_json FROM runtime_events
                WHERE tenant_id = ? AND run_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (tenant_id, run_id, limit),
            ).fetchall()
        return [RuntimeEvent.model_validate(json.loads(row["event_json"])) for row in rows]

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT event_json FROM runtime_events
                WHERE tenant_id = ? AND task_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (tenant_id, task_id, limit),
            ).fetchall()
        return [RuntimeEvent.model_validate(json.loads(row["event_json"])) for row in rows]
