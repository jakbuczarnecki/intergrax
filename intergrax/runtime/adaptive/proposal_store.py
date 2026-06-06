# © Artur Czarnecki. All rights reserved.

"""Persistence for adaptation engine runs (Phase W-ADAPT-2.11)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineRunResult


class ProposalStore(Protocol):
    """Append-only store for adaptation engine runs."""

    def append_run(self, result: AdaptationEngineRunResult) -> None: ...

    def list_runs(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AdaptationEngineRunResult]: ...

    def clear(self) -> None: ...


def default_proposal_store_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[3]
    return root / "build" / "adaptive_harness" / "proposals.db"


class InMemoryProposalStore:
    """In-process proposal run store for unit tests."""

    def __init__(self) -> None:
        self._runs: list[AdaptationEngineRunResult] = []

    def append_run(self, result: AdaptationEngineRunResult) -> None:
        self._runs.append(result)

    def list_runs(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AdaptationEngineRunResult]:
        items = self._runs
        if tenant_id is not None:
            items = [run for run in items if run.tenant_id == tenant_id]
        return items[-limit:]

    def clear(self) -> None:
        self._runs.clear()


class SQLiteProposalStore:
    """SQLite-backed store for adaptation engine runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or default_proposal_store_path()
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptation_engine_runs (
                    run_key TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    task_class TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def append_run(self, result: AdaptationEngineRunResult) -> None:
        run_key = f"{result.tenant_id}:{result.task_class}:{result.generated_at.isoformat()}"
        payload = result.model_dump_json()
        recorded_at = result.generated_at.astimezone(UTC).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO adaptation_engine_runs (
                    run_key, tenant_id, task_class, recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_key, result.tenant_id, result.task_class, recorded_at, payload),
            )

    def list_runs(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[AdaptationEngineRunResult]:
        query = "SELECT payload_json FROM adaptation_engine_runs"
        params: list[str] = []
        if tenant_id is not None:
            query += " WHERE tenant_id = ?"
            params.append(tenant_id)
        query += " ORDER BY recorded_at ASC LIMIT ?"
        params.append(str(max(1, limit)))
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AdaptationEngineRunResult.model_validate_json(row["payload_json"]) for row in rows]

    def clear(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM adaptation_engine_runs")
