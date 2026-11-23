# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite-backed experiment registry (Phase D.3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_EXPERIMENTS_DB,
    ENV_EXPERIMENTS_DB,
    resolve_experiments_db_path,
)
from intergrax.utils.time_provider import SystemTimeProvider

__all__ = [
    "DEFAULT_EXPERIMENTS_DB",
    "ENV_EXPERIMENTS_DB",
    "SQLiteExperimentStore",
    "resolve_experiments_db_path",
]


class SQLiteExperimentStore:
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT '',
                    hypothesis TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    agent_id TEXT,
                    expected_output TEXT NOT NULL DEFAULT '',
                    validation_criteria TEXT NOT NULL DEFAULT '',
                    decision TEXT NOT NULL DEFAULT 'pending',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
            if "tenant_id" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    experiment_id TEXT NOT NULL
                        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                    run_id TEXT NOT NULL,
                    linked_at_utc TEXT NOT NULL,
                    PRIMARY KEY (experiment_id, run_id)
                );
                """
            )

    def register(self, request: RegisterExperimentRequest) -> ExperimentRecord:
        now = SystemTimeProvider.utc_now().isoformat()
        record = ExperimentRecord.new_from_request(
            request,
            created_at_utc=now,
            updated_at_utc=now,
        )
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, tenant_id, hypothesis, capability, agent_id,
                    expected_output, validation_criteria, decision, notes,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.tenant_id,
                    record.hypothesis,
                    record.capability,
                    record.agent_id,
                    record.expected_output,
                    record.validation_criteria,
                    record.decision.value,
                    record.notes,
                    record.created_at_utc,
                    record.updated_at_utc,
                ),
            )
        return record

    def list_experiments(
        self,
        *,
        limit: int = 50,
        decision: ExperimentDecision | None = None,
        tenant_id: str | None = None,
    ) -> list[ExperimentRecord]:
        query = "SELECT * FROM experiments"
        params: list[object] = []
        clauses: list[str] = []
        if decision is not None:
            clauses.append("decision = ?")
            params.append(decision.value)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at_utc DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(conn, row) for row in rows]

    def get(self, experiment_id: str, *, tenant_id: str | None = None) -> ExperimentRecord:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Experiment not found: {experiment_id}")
            if tenant_id is not None and row["tenant_id"] != tenant_id:
                raise ValueError(f"Experiment not found: {experiment_id}")
            return self._row_to_record(conn, row)

    def set_decision(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
        *,
        notes: str | None = None,
        tenant_id: str | None = None,
    ) -> ExperimentRecord:
        now = SystemTimeProvider.utc_now().isoformat()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT notes, tenant_id FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Experiment not found: {experiment_id}")
            if tenant_id is not None and row["tenant_id"] != tenant_id:
                raise ValueError(f"Experiment not found: {experiment_id}")

            merged_notes = row["notes"] or ""
            if notes is not None and notes.strip():
                merged_notes = notes.strip()

            conn.execute(
                """
                UPDATE experiments
                SET decision = ?, notes = ?, updated_at_utc = ?
                WHERE experiment_id = ?
                """,
                (decision.value, merged_notes, now, experiment_id),
            )

            if decision == ExperimentDecision.DELETE:
                conn.execute(
                    "DELETE FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                )
                return ExperimentRecord(
                    experiment_id=experiment_id,
                    tenant_id=row["tenant_id"],
                    hypothesis="",
                    capability="",
                    decision=decision,
                    notes=merged_notes,
                    run_ids=[],
                    created_at_utc=now,
                    updated_at_utc=now,
                )

        return self.get(experiment_id, tenant_id=tenant_id)

    def link_run(
        self,
        experiment_id: str,
        run_id: str,
        *,
        tenant_id: str | None = None,
    ) -> ExperimentRecord:
        now = SystemTimeProvider.utc_now().isoformat()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT tenant_id FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Experiment not found: {experiment_id}")
            if tenant_id is not None and row["tenant_id"] != tenant_id:
                raise ValueError(f"Experiment not found: {experiment_id}")

            conn.execute(
                """
                INSERT OR IGNORE INTO experiment_runs (experiment_id, run_id, linked_at_utc)
                VALUES (?, ?, ?)
                """,
                (experiment_id, run_id, now),
            )
            conn.execute(
                "UPDATE experiments SET updated_at_utc = ? WHERE experiment_id = ?",
                (now, experiment_id),
            )
        return self.get(experiment_id, tenant_id=tenant_id)

    def _row_to_record(self, conn: sqlite3.Connection, row: sqlite3.Row) -> ExperimentRecord:
        run_rows = conn.execute(
            """
            SELECT run_id FROM experiment_runs
            WHERE experiment_id = ?
            ORDER BY linked_at_utc ASC
            """,
            (row["experiment_id"],),
        ).fetchall()
        tenant_key = "tenant_id" if "tenant_id" in row.keys() else None
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            tenant_id=row[tenant_key] if tenant_key else "",
            hypothesis=row["hypothesis"],
            capability=row["capability"],
            agent_id=row["agent_id"],
            expected_output=row["expected_output"],
            validation_criteria=row["validation_criteria"],
            decision=ExperimentDecision(row["decision"]),
            notes=row["notes"],
            run_ids=[r["run_id"] for r in run_rows],
            created_at_utc=row["created_at_utc"],
            updated_at_utc=row["updated_at_utc"],
        )
