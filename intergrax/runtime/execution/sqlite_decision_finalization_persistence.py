# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite durable Decision finalization persistence (DS-REC-01).

Single-host transactional uniqueness via PRIMARY KEY and BEGIN IMMEDIATE.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    DecisionFinalizeGuardState,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_durable_wire_codec import (
    decode_outcome_blob,
    encode_outcome_blob,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionDurableFinalizationResult,
    evaluate_durable_finalization_guard,
)


def _finalization_key_row(key: DecisionFinalizationKey) -> tuple[str, str, str, str]:
    return (
        key.tenant_id,
        str(key.decision_id),
        key.scope.namespace,
        key.scope.subject,
    )


class SQLiteDecisionFinalizationPersistence:
    """Durable single-host finalization store with atomic commit semantics."""

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
                CREATE TABLE IF NOT EXISTS decision_finalizations (
                    tenant_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    scope_namespace TEXT NOT NULL,
                    scope_subject TEXT NOT NULL,
                    outcome_blob TEXT NOT NULL,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject
                    )
                );
                """,
            )

    def _load_existing_guard(
        self,
        conn: sqlite3.Connection,
        key: DecisionFinalizationKey,
    ) -> DecisionFinalizeGuardState[object] | None:
        row = conn.execute(
            """
            SELECT outcome_blob
            FROM decision_finalizations
            WHERE tenant_id = ? AND decision_id = ?
              AND scope_namespace = ? AND scope_subject = ?
            """,
            _finalization_key_row(key),
        ).fetchone()
        if row is None:
            return None
        outcome = decode_outcome_blob(
            row["outcome_blob"],
            payload_codecs=self._payload_codecs,
        )
        return DecisionFinalizeGuardState(key=key, authoritative_outcome=outcome)

    def load_guard_state(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionFinalizeGuardState[object] | None:
        with self._connection() as conn:
            return self._load_existing_guard(conn, key)

    def commit_authoritative_outcome(
        self,
        *,
        key: DecisionFinalizationKey,
        requested_outcome: AuthoritativeAcceptedDecision[object] | AuthoritativeResolutionRecord,
    ) -> DecisionDurableFinalizationResult[object]:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._load_existing_guard(conn, key)
            result = evaluate_durable_finalization_guard(
                existing,
                key=key,
                requested_outcome=requested_outcome,
            )
            if result.disposition is DecisionDurableFinalizationDisposition.COMMITTED:
                outcome = result.guard_state.authoritative_outcome
                if outcome is None:
                    conn.rollback()
                    raise RuntimeError("committed finalization requires authoritative outcome")
                conn.execute(
                    """
                    INSERT INTO decision_finalizations (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject,
                        outcome_blob
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        *_finalization_key_row(key),
                        encode_outcome_blob(
                            outcome,
                            payload_codecs=self._payload_codecs,
                        ),
                    ),
                )
            conn.commit()
            return result
