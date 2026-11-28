# © Artur Czarnecki. All rights reserved.

"""SQLite-backed ADAS candidate archive (AHI-ADAS-10)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
    freeze_candidate_record,
    freeze_evidence_bundle,
    require_archive_read_scope,
    validate_evidence_bundle_against_archive,
    validate_new_selection,
    validate_parent_link,
    validate_transition,
)
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AgentCandidateEvidenceBundle,
    AgentCandidateRecord,
    AgentCandidateStatus,
    compute_evidence_bundle_digest,
    verify_evidence_bundle_digest,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS adas_candidate_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    objective_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    parent_candidate_id TEXT,
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (candidate_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_adas_snapshots_scope
    ON adas_candidate_snapshots (tenant_id, workspace_id, objective_id, candidate_id);

CREATE TABLE IF NOT EXISTS adas_evidence_bundles (
    bundle_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _rollback_if_active(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def _assert_record_write_scope(record: AgentCandidateRecord, *, tenant_id: str, workspace_id: str) -> None:
    if record.tenant_id != tenant_id or record.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("record scope mismatch")
    if record.envelope.tenant_id != tenant_id or record.envelope.workspace_id != workspace_id:
        raise AgentDesignArchiveScopeError("record scope mismatch")


class SQLiteAgentDesignArchive:
    """Durable append-only archive using stdlib sqlite3."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path, isolation_level=None)
            self._connection.row_factory = sqlite3.Row
            self._connection.executescript(_SCHEMA)
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLiteAgentDesignArchive:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _load_scoped_candidate_history(
        self,
        conn: sqlite3.Connection,
        *,
        candidate_id: str,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        rows = conn.execute(
            """
            SELECT payload_json FROM adas_candidate_snapshots
            WHERE candidate_id = ? AND tenant_id = ? AND workspace_id = ?
            ORDER BY sequence, created_at, snapshot_id
            """,
            (candidate_id, tenant_id, workspace_id),
        ).fetchall()
        return [AgentCandidateRecord.model_validate_json(row["payload_json"]) for row in rows]

    def _lineage_history_map_for_parent_validation(
        self,
        conn: sqlite3.Connection,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> dict[str, list[AgentCandidateRecord]]:
        histories: dict[str, list[AgentCandidateRecord]] = {}
        current: str | None = record.parent_candidate_id
        while current is not None:
            if current not in histories:
                histories[current] = self._load_scoped_candidate_history(
                    conn,
                    candidate_id=current,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                )
            parent_hist = histories.get(current)
            if not parent_hist:
                break
            current = parent_hist[-1].parent_candidate_id
        return histories

    def _fetch_scoped_snapshot(
        self,
        conn: sqlite3.Connection,
        *,
        snapshot_id: str,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        row = conn.execute(
            """
            SELECT payload_json FROM adas_candidate_snapshots
            WHERE snapshot_id = ? AND tenant_id = ? AND workspace_id = ?
            """,
            (snapshot_id, tenant_id, workspace_id),
        ).fetchone()
        if row is None:
            return None
        return AgentCandidateRecord.model_validate_json(row["payload_json"])

    def _fetch_prior_snapshot(
        self,
        conn: sqlite3.Connection,
        *,
        candidate_id: str,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        row = conn.execute(
            """
            SELECT payload_json FROM adas_candidate_snapshots
            WHERE candidate_id = ? AND tenant_id = ? AND workspace_id = ?
            ORDER BY sequence DESC, created_at DESC, snapshot_id DESC
            LIMIT 1
            """,
            (candidate_id, tenant_id, workspace_id),
        ).fetchone()
        if row is None:
            return None
        return AgentCandidateRecord.model_validate_json(row["payload_json"])

    def append_candidate_snapshot(
        self,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        _assert_record_write_scope(record, tenant_id=tenant_id, workspace_id=workspace_id)
        frozen = freeze_candidate_record(record)
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            scope_row = conn.execute(
                """
                SELECT tenant_id, workspace_id FROM adas_candidate_snapshots
                WHERE candidate_id = ?
                ORDER BY sequence ASC, created_at ASC, snapshot_id ASC
                LIMIT 1
                """,
                (frozen.candidate_id,),
            ).fetchone()
            if scope_row is not None and (
                scope_row["tenant_id"] != tenant_id or scope_row["workspace_id"] != workspace_id
            ):
                raise AgentDesignArchiveScopeError("candidate identity scope mismatch")

            existing_row = conn.execute(
                "SELECT payload_json FROM adas_candidate_snapshots WHERE snapshot_id = ?",
                (frozen.snapshot_id,),
            ).fetchone()
            if existing_row is not None:
                scope_row_existing = conn.execute(
                    """
                    SELECT tenant_id, workspace_id FROM adas_candidate_snapshots
                    WHERE snapshot_id = ?
                    """,
                    (frozen.snapshot_id,),
                ).fetchone()
                if scope_row_existing is not None and (
                    scope_row_existing["tenant_id"] != tenant_id or scope_row_existing["workspace_id"] != workspace_id
                ):
                    _rollback_if_active(conn)
                    raise AgentDesignArchiveScopeError("candidate identity scope mismatch")
                existing = AgentCandidateRecord.model_validate_json(existing_row["payload_json"])
                if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                    _rollback_if_active(conn)
                    return existing
                _rollback_if_active(conn)
                raise AgentDesignArchiveError("conflicting snapshot identity")

            prior = self._fetch_prior_snapshot(
                conn,
                candidate_id=frozen.candidate_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            lineage = self._lineage_history_map_for_parent_validation(
                conn,
                frozen,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            validate_parent_link(lineage, frozen)
            validate_transition(prior, frozen)
            conn.execute(
                """
                INSERT INTO adas_candidate_snapshots (
                    snapshot_id, candidate_id, objective_id, tenant_id, workspace_id,
                    parent_candidate_id, sequence, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frozen.snapshot_id,
                    frozen.candidate_id,
                    frozen.objective_id,
                    frozen.tenant_id,
                    frozen.workspace_id,
                    frozen.parent_candidate_id,
                    frozen.sequence,
                    frozen.status.value,
                    frozen.model_dump_json(),
                    frozen.created_at.isoformat(),
                ),
            )
            conn.execute("COMMIT")
            committed = True
        except sqlite3.IntegrityError as exc:
            _rollback_if_active(conn)
            existing_row = conn.execute(
                "SELECT payload_json FROM adas_candidate_snapshots WHERE snapshot_id = ?",
                (frozen.snapshot_id,),
            ).fetchone()
            if existing_row is not None:
                scope_row_existing = conn.execute(
                    """
                    SELECT tenant_id, workspace_id FROM adas_candidate_snapshots
                    WHERE snapshot_id = ?
                    """,
                    (frozen.snapshot_id,),
                ).fetchone()
                if scope_row_existing is not None and (
                    scope_row_existing["tenant_id"] != tenant_id or scope_row_existing["workspace_id"] != workspace_id
                ):
                    _rollback_if_active(conn)
                    raise AgentDesignArchiveScopeError("candidate identity scope mismatch")
                existing = AgentCandidateRecord.model_validate_json(existing_row["payload_json"])
                if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                    return existing
            raise AgentDesignArchiveError("conflicting snapshot identity") from exc
        except Exception:
            if not committed:
                _rollback_if_active(conn)
            raise
        return frozen

    def get_latest_candidate_snapshot(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        conn = self._connect()
        return self._fetch_prior_snapshot(
            conn,
            candidate_id=candidate_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def get_candidate_history(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT payload_json FROM adas_candidate_snapshots
            WHERE candidate_id = ? AND tenant_id = ? AND workspace_id = ?
            ORDER BY sequence, created_at, snapshot_id
            """,
            (candidate_id, tenant_id, workspace_id),
        ).fetchall()
        if not rows:
            return []
        return [AgentCandidateRecord.model_validate_json(row["payload_json"]) for row in rows]

    def list_candidates_for_objective(
        self,
        objective_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
        status: AgentCandidateStatus | None = None,
    ) -> list[AgentCandidateRecord]:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT payload_json FROM adas_candidate_snapshots
            WHERE objective_id = ? AND tenant_id = ? AND workspace_id = ?
            ORDER BY candidate_id, sequence, created_at, snapshot_id
            """,
            (objective_id, tenant_id, workspace_id),
        ).fetchall()
        latest: dict[str, AgentCandidateRecord] = {}
        for row in rows:
            record = AgentCandidateRecord.model_validate_json(row["payload_json"])
            prior = latest.get(record.candidate_id)
            if prior is None or record.sequence > prior.sequence:
                latest[record.candidate_id] = record
        values = sorted(latest.values(), key=lambda item: (item.created_at, item.candidate_id, item.sequence))
        if status is None:
            return values
        return [item for item in values if item.status is status]

    def get_child_candidates(
        self,
        parent_candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        parent = self.get_latest_candidate_snapshot(parent_candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)
        if parent is None:
            return []
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT payload_json FROM (
                SELECT candidate_id, MAX(sequence) AS max_seq
                FROM adas_candidate_snapshots
                WHERE tenant_id = ? AND workspace_id = ? AND parent_candidate_id = ?
                GROUP BY candidate_id
            ) latest
            JOIN adas_candidate_snapshots s
              ON s.candidate_id = latest.candidate_id AND s.sequence = latest.max_seq
            """,
            (tenant_id, workspace_id, parent_candidate_id),
        ).fetchall()
        return sorted(
            [AgentCandidateRecord.model_validate_json(row["payload_json"]) for row in rows],
            key=lambda item: item.candidate_id,
        )

    def get_parent_candidate(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        latest = self.get_latest_candidate_snapshot(candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)
        if latest is None or latest.parent_candidate_id is None:
            return None
        return self.get_latest_candidate_snapshot(
            latest.parent_candidate_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def append_sealed_evidence_bundle(
        self,
        bundle: AgentCandidateEvidenceBundle,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("bundle scope mismatch")
        if not bundle.sealed:
            raise AgentDesignArchiveError("bundle must be sealed before append")
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            archived = self._fetch_scoped_snapshot(
                conn,
                snapshot_id=bundle.candidate_record.snapshot_id,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            validate_evidence_bundle_against_archive(bundle, archived)
            digest = bundle.bundle_digest or compute_evidence_bundle_digest(bundle)
            payload = bundle.model_dump(mode="python")
            payload["bundle_digest"] = digest
            frozen = freeze_evidence_bundle(AgentCandidateEvidenceBundle.model_validate(payload))
            verify_evidence_bundle_digest(frozen)
            existing_scope = conn.execute(
                """
                SELECT tenant_id, workspace_id
                FROM adas_evidence_bundles
                WHERE bundle_id = ?
                """,
                (frozen.bundle_id,),
            ).fetchone()
            if existing_scope is not None and (
                existing_scope["tenant_id"] != tenant_id or existing_scope["workspace_id"] != workspace_id
            ):
                _rollback_if_active(conn)
                raise AgentDesignArchiveScopeError("bundle identity scope mismatch")
            existing_row = conn.execute(
                """
                SELECT payload_json
                FROM adas_evidence_bundles
                WHERE bundle_id = ?
                  AND tenant_id = ?
                  AND workspace_id = ?
                """,
                (frozen.bundle_id, tenant_id, workspace_id),
            ).fetchone()
            if existing_row is not None:
                existing = AgentCandidateEvidenceBundle.model_validate_json(existing_row["payload_json"])
                if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                    _rollback_if_active(conn)
                    return existing
                _rollback_if_active(conn)
                raise AgentDesignArchiveError("conflicting evidence bundle identity")
            validate_new_selection(
                frozen,
                self._fetch_prior_snapshot(
                    conn,
                    candidate_id=frozen.candidate_id,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO adas_evidence_bundles (bundle_id, tenant_id, workspace_id, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    frozen.bundle_id,
                    frozen.tenant_id,
                    frozen.workspace_id,
                    frozen.model_dump_json(),
                    frozen.created_at.isoformat(),
                ),
            )
            conn.execute("COMMIT")
            committed = True
        except sqlite3.IntegrityError as exc:
            _rollback_if_active(conn)
            raise AgentDesignArchiveError("conflicting evidence bundle identity") from exc
        except Exception:
            if not committed:
                _rollback_if_active(conn)
            raise
        return frozen

    def get_sealed_evidence_bundle(
        self,
        bundle_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle | None:
        require_archive_read_scope(tenant_id=tenant_id, workspace_id=workspace_id)
        conn = self._connect()
        row = conn.execute(
            """
            SELECT payload_json FROM adas_evidence_bundles
            WHERE bundle_id = ? AND tenant_id = ? AND workspace_id = ?
            """,
            (bundle_id, tenant_id, workspace_id),
        ).fetchone()
        if row is None:
            return None
        try:
            bundle = AgentCandidateEvidenceBundle.model_validate_json(row["payload_json"])
        except ValidationError as exc:
            raise AgentDesignArchiveError("persisted bundle integrity violation") from exc
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            return None
        if bundle.bundle_digest is None:
            raise AgentDesignArchiveError("persisted bundle missing digest")
        verify_evidence_bundle_digest(bundle)
        return bundle
