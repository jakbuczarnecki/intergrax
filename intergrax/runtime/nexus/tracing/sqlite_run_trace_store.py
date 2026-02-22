# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.tracing.persistence_models import (
    RunError,
    RunStats,
    RunTraceWriter,
    RunTraceReader,
    PersistedRun,
    RunMetadata,
    SerializedTraceEvent,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent


class SQLiteRunTraceStore(RunTraceWriter, RunTraceReader):
    """
    Minimal production-grade SQLite-backed trace store.

    Responsibilities:
    - Persist run metadata
    - Persist ordered run events
    - Reconstruct PersistedRun for replay

    This is a P0 minimal implementation.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path: Path = db_path
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


    def _ensure_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    error_json TEXT
                );
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    ts_utc TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                """
            )



    def append_event(self, event: TraceEvent) -> None:
        """
        Persist single trace event in append-only manner.
        """

        serialized = SerializedTraceEvent.from_trace_event(event)
        event_json: str = json.dumps(asdict(serialized))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO run_events (run_id, seq, ts_utc, event_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    serialized.run_id,
                    serialized.seq,
                    serialized.ts_utc,
                    event_json,
                ),
            )

    def finalize_run(self, run_id: str, metadata: RunMetadata) -> None:
        """
        Persist RunMetadata for a completed run.

        Must strictly follow RunTraceWriter contract.
        """

        if run_id != metadata.run_id:
            raise ValueError("run_id mismatch in finalize_run.")

        stats_json: str = json.dumps(asdict(metadata.stats))

        error_json: Optional[str] = None
        if metadata.error is not None:
            error_json = json.dumps(asdict(metadata.error))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    session_id,
                    user_id,
                    tenant_id,
                    started_at_utc,
                    stats_json,
                    error_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.run_id,
                    metadata.session_id,
                    metadata.user_id,
                    metadata.tenant_id,
                    metadata.started_at_utc,
                    stats_json,
                    error_json,
                ),
            )


    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        """
        Reconstruct PersistedRun from SQLite storage.
        """

        with self._get_connection() as conn:
            run_row = conn.execute(
                """
                SELECT run_id,
                       session_id,
                       user_id,
                       tenant_id,
                       started_at_utc,
                       stats_json,
                       error_json
                FROM runs
                WHERE run_id = ?
                AND tenant_id = ?
                """,
                (run_id, tenant_id,),
            ).fetchone()

            if run_row is None:
                raise ValueError(f"Run not found: {run_id}")

            (
                db_run_id,
                session_id,
                user_id,
                tenant_id,
                started_at_utc,
                stats_json,
                error_json,
            ) = run_row

            stats_dict = json.loads(stats_json)
            stats = RunStats(**stats_dict)

            error: Optional[RunError] = None
            if error_json is not None:
                error_dict = json.loads(error_json)
                error = RunError(**error_dict)

            metadata = RunMetadata(
                run_id=db_run_id,
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                started_at_utc=started_at_utc,
                stats=stats,
                error=error,
            )

            event_rows = conn.execute(
                """
                SELECT event_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY seq ASC
                """,
                (run_id,),
            ).fetchall()

            events: List[Dict[str, Any]] = [
                json.loads(row[0]) for row in event_rows
            ]

        return PersistedRun(metadata=metadata, events=events)


