# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite store for human decisions and escalation chain (Phase F.3)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from intergrax.runtime.human.models import (
    EscalationTarget,
    HumanDecisionRecord,
    HumanResponseVerdict,
)
from intergrax.utils.time_provider import SystemTimeProvider

ENV_HUMAN_DECISIONS_DB = "INTERGRAX_HUMAN_DECISIONS_DB"
DEFAULT_HUMAN_DECISIONS_DB = Path("build") / "intergrax_human_decisions.db"


def resolve_human_decisions_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_HUMAN_DECISIONS_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_HUMAN_DECISIONS_DB


def open_human_decision_store(db_path: Path | None = None) -> SQLiteHumanDecisionStore:
    path = db_path or resolve_human_decisions_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteHumanDecisionStore(db_path=path)


class SQLiteHumanDecisionStore:
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
                    created_at_utc TEXT NOT NULL
                );
                """
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
                    agent_id, run_id, notes, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> HumanDecisionRecord:
        target = row["escalation_target"]
        return HumanDecisionRecord(
            decision_id=row["decision_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
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

    @classmethod
    def build_record(
        cls,
        *,
        task_id: str,
        tenant_id: str,
        user_id: str,
        verdict: HumanResponseVerdict,
        response_text: str,
        human_request_id: str = "",
        escalation_level: int = 0,
        escalation_target: Optional[EscalationTarget] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        notes: str = "",
    ) -> HumanDecisionRecord:
        return HumanDecisionRecord(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            human_request_id=human_request_id,
            verdict=verdict,
            response_text=response_text,
            escalation_level=escalation_level,
            escalation_target=escalation_target,
            agent_id=agent_id,
            run_id=run_id,
            notes=notes,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        )
