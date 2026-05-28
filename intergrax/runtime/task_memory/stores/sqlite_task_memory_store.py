# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite implementation of TaskMemoryPersistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence


class SQLiteTaskMemoryStore(TaskMemoryPersistence):
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
                CREATE TABLE IF NOT EXISTS task_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE (tenant_id, task_id, namespace, key)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_memory_task
                ON task_memory (tenant_id, task_id, namespace, key);
                """
            )

    def put(self, record: TaskMemoryRecord) -> TaskMemoryRecord:
        payload = record.model_dump(mode="json")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO task_memory (
                    record_id, tenant_id, task_id, namespace, key,
                    value_json, created_at_utc, updated_at_utc, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, task_id, namespace, key) DO UPDATE SET
                    record_id = excluded.record_id,
                    value_json = excluded.value_json,
                    updated_at_utc = excluded.updated_at_utc,
                    provenance_json = excluded.provenance_json
                """,
                (
                    record.record_id,
                    record.tenant_id,
                    record.task_id,
                    record.namespace,
                    record.key,
                    json.dumps(payload["value"]),
                    record.created_at_utc,
                    record.updated_at_utc,
                    json.dumps(payload.get("provenance") or {}),
                ),
            )
        return record

    def get(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> Optional[TaskMemoryRecord]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT record_id, tenant_id, task_id, namespace, key,
                       value_json, created_at_utc, updated_at_utc, provenance_json
                FROM task_memory
                WHERE tenant_id = ? AND task_id = ? AND namespace = ? AND key = ?
                """,
                (tenant_id, task_id, namespace, key),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def delete(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM task_memory
                WHERE tenant_id = ? AND task_id = ? AND namespace = ? AND key = ?
                """,
                (tenant_id, task_id, namespace, key),
            )
        return cursor.rowcount > 0

    def list_records(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> List[TaskMemoryRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT record_id, tenant_id, task_id, namespace, key,
                       value_json, created_at_utc, updated_at_utc, provenance_json
                FROM task_memory
                WHERE tenant_id = ? AND task_id = ? AND namespace = ?
                  AND key LIKE ?
                ORDER BY key ASC
                LIMIT ?
                """,
                (tenant_id, task_id, namespace, f"{prefix}%", limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count_for_task(self, *, tenant_id: str, task_id: str) -> int:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM task_memory
                WHERE tenant_id = ? AND task_id = ?
                """,
                (tenant_id, task_id),
            ).fetchone()
        return int(row["cnt"]) if row is not None else 0

    def clear_task(self, *, tenant_id: str, task_id: str) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM task_memory
                WHERE tenant_id = ? AND task_id = ?
                """,
                (tenant_id, task_id),
            )
        return cursor.rowcount

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskMemoryRecord:
        return TaskMemoryRecord(
            record_id=str(row["record_id"]),
            tenant_id=str(row["tenant_id"]),
            task_id=str(row["task_id"]),
            namespace=str(row["namespace"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
            provenance=json.loads(row["provenance_json"] or "{}"),
        )
