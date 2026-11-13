# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Durable single-host execution ``ArtifactStore`` on SQLite (PDC-D8-R007, R014).

Follows the canonical SQLite store convention of the persistence plane (``db_path`` constructor,
``PersistenceTopology.DURABLE_SINGLE_HOST``, WAL, ``synchronous=FULL``). Storage-only: no
authorization, no tenant policy, no audit — those live in the governed access port.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime

from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactRetentionClass,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.store_base import (
    ArtifactDeletionRefused,
    ArtifactStore,
    ArtifactVersionNotFound,
    protection_reason,
    require_owned_class,
)

_SCHEMA_VERSION = "execution_artifact.v1"

_COLUMNS = (
    "tenant_id, artifact_id, version, run_id, step_id, kind, mime_type, created_at_utc, data, "
    "size_bytes, content_digest, artifact_class, retention_class, expires_at_utc, workspace_id, "
    "producer_json, lineage_json"
)

_LATEST_PER_ARTIFACT = (
    "version = (SELECT MAX(b.version) FROM execution_artifacts b "
    "WHERE b.tenant_id = execution_artifacts.tenant_id "
    "AND b.artifact_id = execution_artifacts.artifact_id)"
)


def _epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


class SQLiteArtifactStore(ArtifactStore):
    """Append-only, tenant-keyed artifact store that survives a process restart."""

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.DURABLE_SINGLE_HOST

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_schema()

    # ------------------------------------------------------------------ plumbing

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_artifacts (
                    tenant_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    step_id TEXT,
                    kind TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    data BLOB NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    artifact_class TEXT NOT NULL,
                    retention_class TEXT NOT NULL,
                    expires_at_utc TEXT,
                    expires_epoch REAL,
                    workspace_id TEXT,
                    producer_json TEXT,
                    lineage_json TEXT,
                    schema_version TEXT NOT NULL DEFAULT 'execution_artifact.v1',
                    PRIMARY KEY (tenant_id, artifact_id, version)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_artifacts_run ON execution_artifacts (tenant_id, run_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_artifacts_expiry "
                "ON execution_artifacts (tenant_id, expires_epoch)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_artifact_legal_holds (
                    tenant_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    placed_by TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    placed_at_utc TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, artifact_id)
                )
                """
            )

    @staticmethod
    def _to_artifact(row: tuple) -> Artifact:
        (
            tenant_id,
            artifact_id,
            version,
            run_id,
            step_id,
            kind,
            mime_type,
            created_at,
            data,
            size_bytes,
            content_digest,
            artifact_class,
            retention_class,
            expires_at,
            workspace_id,
            producer_json,
            lineage_json,
        ) = row
        # ``Artifact`` re-derives the digest from ``data``: a tampered row fails on load.
        return Artifact(
            tenant_id=tenant_id,
            artifact_id=artifact_id,
            run_id=run_id,
            step_id=step_id,
            kind=kind,
            mime_type=mime_type,
            created_at_utc=datetime.fromisoformat(created_at),
            data=bytes(data),
            size_bytes=size_bytes,
            content_digest=content_digest,
            version=version,
            artifact_class=ArtifactClass(artifact_class),
            retention_class=ArtifactRetentionClass(retention_class),
            expires_at_utc=_parse(expires_at),
            workspace_id=workspace_id,
            producer=(ArtifactProvenance.model_validate_json(producer_json) if producer_json else None),
            lineage=ArtifactLineage.model_validate_json(lineage_json) if lineage_json else None,
        )

    # ------------------------------------------------------------------ writes

    def put(self, artifact: Artifact) -> Artifact:
        require_owned_class(artifact.artifact_class)
        with self._transaction() as conn:
            current = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ?",
                (artifact.tenant_id, artifact.artifact_id),
            ).fetchone()[0]
            stored = replace(artifact, version=current + 1)
            conn.execute(
                f"INSERT INTO execution_artifacts ({_COLUMNS}, expires_epoch, schema_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    stored.tenant_id,
                    stored.artifact_id,
                    stored.version,
                    stored.run_id,
                    stored.step_id,
                    stored.kind,
                    stored.mime_type,
                    stored.created_at_utc.isoformat(),
                    stored.data,
                    stored.size_bytes,
                    stored.content_digest,
                    stored.artifact_class.value,
                    stored.retention_class.value,
                    _iso(stored.expires_at_utc),
                    stored.workspace_id,
                    stored.producer.model_dump_json() if stored.producer else None,
                    stored.lineage.model_dump_json() if stored.lineage else None,
                    None if stored.expires_at_utc is None else _epoch(stored.expires_at_utc),
                    _SCHEMA_VERSION,
                ),
            )
            return stored

    # ------------------------------------------------------------------ reads

    def get(self, tenant_id: str, artifact_id: str) -> Artifact:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM execution_artifacts "
                "WHERE tenant_id = ? AND artifact_id = ? ORDER BY version DESC LIMIT 1",
                (tenant_id, artifact_id),
            ).fetchone()
        if row is None:
            raise KeyError((tenant_id, artifact_id))
        return self._to_artifact(row)

    def get_version(self, tenant_id: str, artifact_id: str, version: int) -> Artifact:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ? AND version = ?",
                (tenant_id, artifact_id, version),
            ).fetchone()
        if row is None:
            raise ArtifactVersionNotFound((tenant_id, artifact_id, version))
        return self._to_artifact(row)

    def list_versions(self, tenant_id: str, artifact_id: str) -> Iterable[Artifact]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM execution_artifacts "
                "WHERE tenant_id = ? AND artifact_id = ? ORDER BY version ASC",
                (tenant_id, artifact_id),
            ).fetchall()
        return [self._to_artifact(row) for row in rows]

    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[Artifact]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM execution_artifacts "
                f"WHERE tenant_id = ? AND run_id = ? AND {_LATEST_PER_ARTIFACT} "
                "ORDER BY created_at_utc ASC, artifact_id ASC",
                (tenant_id, run_id),
            ).fetchall()
        return [self._to_artifact(row) for row in rows]

    def list_expired(self, tenant_id: str, now: datetime, *, limit: int = 1000) -> list[Artifact]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM execution_artifacts "
                "WHERE tenant_id = ? AND expires_epoch IS NOT NULL AND expires_epoch <= ? "
                f"AND {_LATEST_PER_ARTIFACT} "
                "ORDER BY expires_epoch ASC, artifact_id ASC LIMIT ?",
                (tenant_id, _epoch(now), limit),
            ).fetchall()
        return [self._to_artifact(row) for row in rows]

    # ------------------------------------------------------------------ deletion and holds

    @staticmethod
    def _refusal(conn: sqlite3.Connection, tenant_id: str, artifact_id: str) -> str | None:
        classes = [
            ArtifactRetentionClass(row[0])
            for row in conn.execute(
                "SELECT DISTINCT retention_class FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ?",
                (tenant_id, artifact_id),
            ).fetchall()
        ]
        held = (
            conn.execute(
                "SELECT 1 FROM execution_artifact_legal_holds WHERE tenant_id = ? AND artifact_id = ?",
                (tenant_id, artifact_id),
            ).fetchone()
            is not None
        )
        return protection_reason(classes, legal_hold=held)

    def delete_artifact(self, tenant_id: str, artifact_id: str) -> None:
        with self._transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ? LIMIT 1",
                (tenant_id, artifact_id),
            ).fetchone()
            if exists is None:
                raise KeyError((tenant_id, artifact_id))
            reason = self._refusal(conn, tenant_id, artifact_id)
            if reason is not None:
                raise ArtifactDeletionRefused(artifact_id, reason)
            conn.execute(
                "DELETE FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ?",
                (tenant_id, artifact_id),
            )

    def delete_for_run(self, tenant_id: str, run_id: str) -> None:
        with self._transaction() as conn:
            targets = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT artifact_id FROM execution_artifacts "
                    f"WHERE tenant_id = ? AND run_id = ? AND {_LATEST_PER_ARTIFACT}",
                    (tenant_id, run_id),
                ).fetchall()
            ]
            for artifact_id in targets:
                reason = self._refusal(conn, tenant_id, artifact_id)
                if reason is not None:
                    raise ArtifactDeletionRefused(artifact_id, reason)
            for artifact_id in targets:
                conn.execute(
                    "DELETE FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ?",
                    (tenant_id, artifact_id),
                )

    def place_legal_hold(self, tenant_id: str, artifact_id: str, *, placed_by: str, reason: str) -> None:
        with self._transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM execution_artifacts WHERE tenant_id = ? AND artifact_id = ? LIMIT 1",
                (tenant_id, artifact_id),
            ).fetchone()
            if exists is None:
                raise KeyError((tenant_id, artifact_id))
            conn.execute(
                "INSERT OR REPLACE INTO execution_artifact_legal_holds "
                "(tenant_id, artifact_id, placed_by, reason, placed_at_utc) VALUES (?,?,?,?,?)",
                (tenant_id, artifact_id, placed_by, reason, datetime.now(UTC).isoformat()),
            )

    def release_legal_hold(self, tenant_id: str, artifact_id: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM execution_artifact_legal_holds WHERE tenant_id = ? AND artifact_id = ?",
                (tenant_id, artifact_id),
            )

    def is_under_legal_hold(self, tenant_id: str, artifact_id: str) -> bool:
        with closing(self._connect()) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM execution_artifact_legal_holds WHERE tenant_id = ? AND artifact_id = ?",
                    (tenant_id, artifact_id),
                ).fetchone()
                is not None
            )
