# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite durable Decision checkpoint persistence (DS-REC-02)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.runtime.execution.decision_durable_wire_codec import (
    decode_checkpoint_blob,
    encode_checkpoint_blob,
)


def _checkpoint_key_row(key: DecisionFinalizationKey) -> tuple[str, str, str, str]:
    return (
        key.tenant_id,
        str(key.decision_id),
        key.scope.namespace,
        key.scope.subject,
    )


class SQLiteDecisionCheckpointPersistence:
    """Durable single-host checkpoint store keyed by finalization scope."""

    __slots__ = ("_db_path",)

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_checkpoints (
                    tenant_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    scope_namespace TEXT NOT NULL,
                    scope_subject TEXT NOT NULL,
                    checkpoint_blob TEXT NOT NULL,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject
                    )
                );
                """,
            )

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[object] | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT checkpoint_blob
                FROM decision_checkpoints
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                """,
                _checkpoint_key_row(key),
            ).fetchone()
        if row is None:
            return None
        checkpoint = decode_checkpoint_blob(row["checkpoint_blob"])
        return restore_decision_checkpoint_state(checkpoint)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[object],
    ) -> None:
        validated = restore_decision_checkpoint_state(checkpoint)
        key = validated.finalization.key
        blob = encode_checkpoint_blob(validated)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO decision_checkpoints (
                    tenant_id,
                    decision_id,
                    scope_namespace,
                    scope_subject,
                    checkpoint_blob
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, decision_id, scope_namespace, scope_subject)
                DO UPDATE SET checkpoint_blob = excluded.checkpoint_blob
                """,
                (*_checkpoint_key_row(key), blob),
            )
            conn.commit()
