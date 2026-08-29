# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite implementation of RuntimeEventPersistence (Tier-0 default backend)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

from intergrax.runtime.events.execution_position import (
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    _validate_through_limit,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, parse_runtime_event_payload


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
                    event_json TEXT NOT NULL,
                    execution_position INTEGER
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_event_run_sequences (
                    tenant_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    next_position INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, run_id)
                );
                """
            )
            self._migrate_execution_position(conn)
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_runtime_events_run_position
                ON runtime_events (tenant_id, run_id, execution_position);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_events_run
                ON runtime_events (tenant_id, run_id, execution_position ASC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_events_task
                ON runtime_events (tenant_id, task_id, execution_position ASC);
                """
            )

    def _migrate_execution_position(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runtime_events)").fetchall()
        }
        if "execution_position" not in columns:
            conn.execute(
                "ALTER TABLE runtime_events ADD COLUMN execution_position INTEGER"
            )
        rows = conn.execute(
            """
            SELECT tenant_id, run_id, event_id, id
            FROM runtime_events
            WHERE execution_position IS NULL
            ORDER BY tenant_id, run_id, id ASC
            """
        ).fetchall()
        counters: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row["tenant_id"], row["run_id"])
            position = counters.get(key, 1)
            conn.execute(
                """
                UPDATE runtime_events
                SET execution_position = ?
                WHERE event_id = ?
                """,
                (position, row["event_id"]),
            )
            counters[key] = position + 1
        for (tenant_id, run_id), next_position in counters.items():
            conn.execute(
                """
                INSERT INTO runtime_event_run_sequences (tenant_id, run_id, next_position)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id, run_id) DO UPDATE SET
                    next_position = excluded.next_position
                """,
                (tenant_id, run_id, next_position),
            )

    def _load_positioned(self, row: sqlite3.Row) -> PositionedRuntimeEvent:
        raw_position = row["execution_position"]
        if raw_position is None:
            raise RuntimeError("runtime_events row missing execution_position")
        return PositionedRuntimeEvent(
            event=parse_runtime_event_payload(json.loads(row["event_json"])),
            position=ExecutionEventPosition(int(raw_position)),
        )

    def _allocate_position(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        run_id: str,
    ) -> ExecutionEventPosition:
        row = conn.execute(
            """
            INSERT INTO runtime_event_run_sequences (tenant_id, run_id, next_position)
            VALUES (?, ?, 2)
            ON CONFLICT(tenant_id, run_id) DO UPDATE SET
                next_position = runtime_event_run_sequences.next_position + 1
            RETURNING next_position - 1 AS allocated_position
            """,
            (tenant_id, run_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to allocate execution position")
        return ExecutionEventPosition(int(row["allocated_position"]))

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        scope = tenant_id or event.tenant_id or ""
        payload = event.model_dump(mode="json")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    """
                    SELECT event_json, execution_position
                    FROM runtime_events
                    WHERE event_id = ?
                    """,
                    (event.event_id,),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return self._load_positioned(existing)
                position = self._allocate_position(
                    conn,
                    tenant_id=scope,
                    run_id=event.run_id,
                )
                conn.execute(
                    """
                    INSERT INTO runtime_events (
                        tenant_id, task_id, run_id, event_id, event_type,
                        created_at_utc, event_json, execution_position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scope,
                        event.task_id,
                        event.run_id,
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        json.dumps(payload),
                        position.value,
                    ),
                )
                conn.commit()
                return PositionedRuntimeEvent(event=event, position=position)
            except Exception:
                conn.rollback()
                raise

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        limit, through = _validate_through_limit(limit=limit, through=through)
        query = """
            SELECT event_json, execution_position
            FROM runtime_events
            WHERE tenant_id = ? AND run_id = ?
        """
        params: list[object] = [tenant_id, run_id]
        if through is not None:
            query += " AND execution_position <= ?"
            params.append(through.value)
        query += " ORDER BY execution_position ASC LIMIT ?"
        params.append(limit)
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._load_positioned(row) for row in rows]

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be > 0")
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT event_json, execution_position
                FROM runtime_events
                WHERE tenant_id = ? AND task_id = ?
                ORDER BY execution_position ASC
                LIMIT ?
                """,
                (tenant_id, task_id, limit),
            ).fetchall()
        return [self._load_positioned(row).event for row in rows]
