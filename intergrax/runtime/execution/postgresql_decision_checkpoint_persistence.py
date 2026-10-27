# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""PostgreSQL durable Decision checkpoint persistence (DIST-SCALE-X P4-A / SX-F03)."""

from __future__ import annotations

import hashlib
import struct

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    is_postgresql_unique_violation,
)
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    MaterializedDecisionCheckpoint,
    StaleDecisionCheckpointWriteError,
)
from intergrax.runtime.execution.decision_durable_wire_codec import (
    decode_checkpoint_blob,
    encode_checkpoint_blob,
)
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    _checkpoint_key_row,
)


class PostgreSQLDecisionCheckpointPersistence:
    """Durable decision checkpoint store keyed by finalization scope (PG parity with SQLite)."""

    __slots__ = ("_provider", "_payload_codecs")

    def __init__(
        self,
        *,
        config: PostgreSQLIntegrationConfig | None = None,
        provider: PostgreSQLConnectionProvider | None = None,
        payload_codecs: DecisionArtifactPayloadCodecRegistry,
    ) -> None:
        if provider is None:
            resolved = config or PostgreSQLIntegrationConfig.from_env()
            provider = PostgreSQLConnectionProvider(resolved)
        self._provider = provider
        self._payload_codecs = payload_codecs
        self._ensure_schema()

    def _schema_init_advisory_key(self) -> int:
        digest = hashlib.sha256(f"ddl-f03:{self._provider.tenant_schema}".encode()).digest()
        return struct.unpack("!q", digest[:8])[0]

    def _ensure_schema(self) -> None:
        with self._provider.transaction() as session:
            session.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self._schema_init_advisory_key(),),
            )
            session.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_checkpoints (
                    tenant_id TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    scope_namespace TEXT NOT NULL,
                    scope_subject TEXT NOT NULL,
                    checkpoint_blob TEXT NOT NULL,
                    snapshot_revision INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (
                        tenant_id,
                        decision_id,
                        scope_namespace,
                        scope_subject
                    )
                );
                """,
            )

    def _fetch_row(
        self,
        session,
        key: DecisionFinalizationKey,
    ):
        return session.execute(
            """
            SELECT checkpoint_blob, snapshot_revision
            FROM decision_checkpoints
            WHERE tenant_id = %s AND decision_id = %s
              AND scope_namespace = %s AND scope_subject = %s
            """,
            _checkpoint_key_row(key),
        ).fetchone()

    def load_materialized(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> MaterializedDecisionCheckpoint[object] | None:
        with self._provider.connection() as session:
            row = self._fetch_row(session, key)
        if row is None:
            return None
        revision = int(row["snapshot_revision"])
        if revision < 1:
            return None
        checkpoint = decode_checkpoint_blob(
            row["checkpoint_blob"],
            payload_codecs=self._payload_codecs,
        )
        restored = restore_decision_checkpoint_state(checkpoint)
        return MaterializedDecisionCheckpoint(
            key=key,
            checkpoint=restored,
            snapshot_revision=revision,
        )

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[object] | None:
        materialized = self.load_materialized(key=key)
        if materialized is None:
            return None
        return materialized.checkpoint

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
        with self._provider.transaction() as session:
            row = self._fetch_row(session, key)
            if expected_revision is None:
                if row is None:
                    try:
                        session.execute(
                            """
                            INSERT INTO decision_checkpoints (
                                tenant_id, decision_id, scope_namespace, scope_subject,
                                checkpoint_blob, snapshot_revision
                            ) VALUES (%s, %s, %s, %s, %s, 1)
                            """,
                            (*key_row, blob),
                        )
                    except Exception as exc:
                        if is_postgresql_unique_violation(exc):
                            row = self._fetch_row(session, key)
                            if row is None:
                                raise
                            session.execute(
                                """
                                UPDATE decision_checkpoints
                                SET checkpoint_blob = %s
                                WHERE tenant_id = %s AND decision_id = %s
                                  AND scope_namespace = %s AND scope_subject = %s
                                """,
                                (blob, *key_row),
                            )
                        else:
                            raise
                else:
                    session.execute(
                        """
                        UPDATE decision_checkpoints
                        SET checkpoint_blob = %s
                        WHERE tenant_id = %s AND decision_id = %s
                          AND scope_namespace = %s AND scope_subject = %s
                        """,
                        (blob, *key_row),
                    )
                return

            current_revision = int(row["snapshot_revision"]) if row is not None else 0
            if current_revision != expected_revision:
                raise StaleDecisionCheckpointWriteError(
                    f"expected snapshot_revision={expected_revision}, "
                    f"actual={current_revision}",
                )
            if row is None:
                session.execute(
                    """
                    INSERT INTO decision_checkpoints (
                        tenant_id, decision_id, scope_namespace, scope_subject,
                        checkpoint_blob, snapshot_revision
                    ) VALUES (%s, %s, %s, %s, %s, 1)
                    """,
                    (*key_row, blob),
                )
            else:
                updated = session.execute(
                    """
                    UPDATE decision_checkpoints
                    SET checkpoint_blob = %s,
                        snapshot_revision = snapshot_revision + 1
                    WHERE tenant_id = %s AND decision_id = %s
                      AND scope_namespace = %s AND scope_subject = %s
                      AND snapshot_revision = %s
                    """,
                    (blob, *key_row, expected_revision),
                )
                if updated.rowcount != 1:
                    raise StaleDecisionCheckpointWriteError(
                        "concurrent decision checkpoint snapshot CAS lost",
                    )
