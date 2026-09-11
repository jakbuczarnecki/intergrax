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
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    StaleDecisionCheckpointWriteError,
)
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

    __slots__ = ("_db_path", "_payload_codecs")

    def __init__(
        self,
        *,
        db_path: Path,
        payload_codecs: DecisionArtifactPayloadCodecRegistry,
    ) -> None:
        self._db_path = db_path
        self._payload_codecs = payload_codecs
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
                    snapshot_revision INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject
                    )
                );
                """,
            )
            self._migrate_snapshot_revision_column(conn)

    def _migrate_snapshot_revision_column(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(decision_checkpoints)").fetchall()
        }
        if "snapshot_revision" not in columns:
            conn.execute(
                "ALTER TABLE decision_checkpoints ADD COLUMN snapshot_revision INTEGER NOT NULL DEFAULT 0",
            )

    def materialized_revision(self, *, key: DecisionFinalizationKey) -> int:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT snapshot_revision
                FROM decision_checkpoints
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                """,
                _checkpoint_key_row(key),
            ).fetchone()
        if row is None:
            return 0
        return int(row["snapshot_revision"])

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
        checkpoint = decode_checkpoint_blob(
            row["checkpoint_blob"],
            payload_codecs=self._payload_codecs,
        )
        return restore_decision_checkpoint_state(checkpoint)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[object],
        expected_revision: int | None = None,
    ) -> None:
        validated = restore_decision_checkpoint_state(checkpoint)
        key = validated.finalization.key
        blob = encode_checkpoint_blob(
            validated,
            payload_codecs=self._payload_codecs,
        )
        key_row = _checkpoint_key_row(key)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT snapshot_revision
                FROM decision_checkpoints
                WHERE tenant_id = ? AND decision_id = ?
                  AND scope_namespace = ? AND scope_subject = ?
                """,
                key_row,
            ).fetchone()
            if expected_revision is None:
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO decision_checkpoints (
                            tenant_id, decision_id, scope_namespace, scope_subject,
                            checkpoint_blob, snapshot_revision
                        ) VALUES (?, ?, ?, ?, ?, 1)
                        """,
                        (*key_row, blob),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE decision_checkpoints
                        SET checkpoint_blob = ?
                        WHERE tenant_id = ? AND decision_id = ?
                          AND scope_namespace = ? AND scope_subject = ?
                        """,
                        (blob, *key_row),
                    )
                conn.commit()
                return

            current_revision = int(row["snapshot_revision"]) if row is not None else 0
            if current_revision != expected_revision:
                conn.rollback()
                raise StaleDecisionCheckpointWriteError(
                    f"expected snapshot_revision={expected_revision}, "
                    f"actual={current_revision}",
                )
            if row is None:
                conn.execute(
                    """
                    INSERT INTO decision_checkpoints (
                        tenant_id, decision_id, scope_namespace, scope_subject,
                        checkpoint_blob, snapshot_revision
                    ) VALUES (?, ?, ?, ?, ?, 1)
                    """,
                    (*key_row, blob),
                )
            else:
                updated = conn.execute(
                    """
                    UPDATE decision_checkpoints
                    SET checkpoint_blob = ?,
                        snapshot_revision = snapshot_revision + 1
                    WHERE tenant_id = ? AND decision_id = ?
                      AND scope_namespace = ? AND scope_subject = ?
                      AND snapshot_revision = ?
                    """,
                    (blob, *key_row, expected_revision),
                )
                if updated.rowcount != 1:
                    conn.rollback()
                    raise StaleDecisionCheckpointWriteError(
                        "concurrent decision checkpoint snapshot CAS lost",
                    )
            conn.commit()
