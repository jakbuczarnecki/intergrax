# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite store for human decisions and escalation chain (Phase F.3)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_HUMAN_DECISIONS_DB,
    ENV_HUMAN_DECISIONS_DB,
    resolve_human_decisions_db_path,
)
from intergrax.contracts.human_approver import (
    HumanApproverEvidence,
    local_development_approver_evidence,
)
from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanDecisionRecord,
    HumanResponseVerdict,
)
from intergrax.runtime.human.persistence_contract import HumanDecisionPersistence

__all__ = [
    "DEFAULT_HUMAN_DECISIONS_DB",
    "ENV_HUMAN_DECISIONS_DB",
    "SQLiteHumanDecisionStore",
    "resolve_human_decisions_db_path",
    "open_human_decision_store",
]


def open_human_decision_store(db_path: Path | None = None) -> SQLiteHumanDecisionStore:
    from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_human_decision_store

    if db_path is not None:
        return create_sqlite_human_decision_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_human_decision_store()  # type: ignore[return-value]


class SQLiteHumanDecisionStore(HumanDecisionPersistence):
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS human_decisions (
                    decision_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    human_request_id TEXT NOT NULL DEFAULT '',
                    verdict TEXT NOT NULL,
                    response_text TEXT NOT NULL DEFAULT '',
                    escalation_level INTEGER NOT NULL DEFAULT 0,
                    escalation_target TEXT,
                    agent_id TEXT,
                    run_id TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at_utc TEXT NOT NULL,
                    approver_json TEXT
                );
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(human_decisions)").fetchall()
            }
            if "approver_json" not in columns:
                conn.execute(
                    "ALTER TABLE human_decisions ADD COLUMN approver_json TEXT"
                )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_human_decisions_task
                ON human_decisions (task_id, tenant_id);
                """
            )

    def record(self, record: HumanDecisionRecord) -> HumanDecisionRecord:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO human_decisions (
                    decision_id, task_id, tenant_id, user_id, human_request_id,
                    verdict, response_text, escalation_level, escalation_target,
                    agent_id, run_id, notes, created_at_utc, approver_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.decision_id,
                    record.task_id,
                    record.tenant_id,
                    record.user_id,
                    record.human_request_id,
                    record.verdict.value,
                    record.response_text,
                    record.escalation_level,
                    record.escalation_target.value if record.escalation_target else None,
                    record.agent_id,
                    record.run_id,
                    record.notes,
                    record.created_at_utc,
                    record.approver.model_dump_json(),
                ),
            )
        return record

    def list_for_task(self, task_id: str, tenant_id: str) -> List[HumanDecisionRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM human_decisions
                WHERE task_id = ? AND tenant_id = ?
                ORDER BY created_at_utc ASC
                """,
                (task_id, tenant_id),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_escalations(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
    ) -> List[HumanDecisionRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM human_decisions
                WHERE tenant_id = ? AND verdict = ?
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (tenant_id, HumanResponseVerdict.ESCALATE.value, limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_decision(self, decision_id: str, tenant_id: str) -> Optional[HumanDecisionRecord]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM human_decisions
                WHERE decision_id = ? AND tenant_id = ?
                """,
                (decision_id, tenant_id),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def summarize_queue(self, tenant_id: str) -> dict[str, int]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT verdict, COUNT(*) AS cnt
                FROM human_decisions
                WHERE tenant_id = ?
                GROUP BY verdict
                """,
                (tenant_id,),
            ).fetchall()
        return {str(row["verdict"]): int(row["cnt"]) for row in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> HumanDecisionRecord:
        target = row["escalation_target"]
        approver_raw = row["approver_json"] if "approver_json" in row.keys() else None
        if approver_raw:
            approver = HumanApproverEvidence.model_validate_json(approver_raw)
        else:
            approver = local_development_approver_evidence(
                tenant_id=row["tenant_id"],
                actor_id=row["user_id"] or "legacy_unknown_approver",
            )
        return HumanDecisionRecord(
            decision_id=row["decision_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
            approver=approver,
            user_id=row["user_id"],
            human_request_id=row["human_request_id"],
            verdict=HumanResponseVerdict(row["verdict"]),
            response_text=row["response_text"],
            escalation_level=int(row["escalation_level"]),
            escalation_target=EscalationTarget(target) if target else None,
            agent_id=row["agent_id"],
            run_id=row["run_id"],
            notes=row["notes"],
            created_at_utc=row["created_at_utc"],
        )

