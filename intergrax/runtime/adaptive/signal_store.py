# © Artur Czarnecki. All rights reserved.

"""Persistence for harness outcome signals (Phase W-ADAPT-1.3)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal


class SignalStore(Protocol):
    """Append-only store for harness outcome signals."""

    def append(self, signal: HarnessOutcomeSignal) -> None: ...

    def list_signals(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[HarnessOutcomeSignal]: ...

    def clear(self) -> None: ...


def default_signal_store_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "signals.db"


class InMemorySignalStore:
    """In-process signal store for unit tests."""

    def __init__(self) -> None:
        self._signals: list[HarnessOutcomeSignal] = []

    def append(self, signal: HarnessOutcomeSignal) -> None:
        self._signals.append(signal)

    def list_signals(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[HarnessOutcomeSignal]:
        filtered = self._signals
        if tenant_id is not None:
            filtered = [item for item in filtered if item.tenant_id == tenant_id]
        if since is not None:
            filtered = [item for item in filtered if item.timestamp >= since]
        if until is not None:
            filtered = [item for item in filtered if item.timestamp <= until]
        return filtered[-limit:]

    def clear(self) -> None:
        self._signals.clear()


class SQLiteSignalStore:
    """SQLite-backed signal store under ``build/adaptive_harness/`` (gitignored)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_signal_store_path()
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harness_outcome_signals (
                    signal_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_harness_outcome_signals_tenant_time
                ON harness_outcome_signals (tenant_id, recorded_at);
                """
            )

    def append(self, signal: HarnessOutcomeSignal) -> None:
        payload = signal.model_dump_json()
        recorded_at = signal.timestamp.astimezone(UTC).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO harness_outcome_signals (
                    signal_id, tenant_id, recorded_at, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (signal.signal_id, signal.tenant_id, recorded_at, payload),
            )

    def list_signals(
        self,
        *,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[HarnessOutcomeSignal]:
        query = "SELECT payload_json FROM harness_outcome_signals"
        clauses: list[str] = []
        params: list[str] = []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if since is not None:
            clauses.append("recorded_at >= ?")
            params.append(since.astimezone(UTC).isoformat())
        if until is not None:
            clauses.append("recorded_at <= ?")
            params.append(until.astimezone(UTC).isoformat())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY recorded_at ASC LIMIT ?"
        params.append(str(max(1, limit)))

        signals: list[HarnessOutcomeSignal] = []
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            signals.append(HarnessOutcomeSignal.model_validate_json(row["payload_json"]))
        return signals

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM harness_outcome_signals")


_default_sqlite_store: SQLiteSignalStore | None = None


def default_signal_store() -> SQLiteSignalStore:
    global _default_sqlite_store
    if _default_sqlite_store is None:
        _default_sqlite_store = SQLiteSignalStore()
    return _default_sqlite_store


def reset_default_signal_store_for_tests() -> None:
    """Clear module singleton between tests."""
    global _default_sqlite_store
    _default_sqlite_store = None
