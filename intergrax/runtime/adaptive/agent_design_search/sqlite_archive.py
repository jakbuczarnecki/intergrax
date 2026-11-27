# © Artur Czarnecki. All rights reserved.

"""SQLite-backed ADAS candidate archive (AHI-ADAS-10)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
    InMemoryAgentDesignArchive,
    _assert_scope,
    _freeze_record,
    _validate_parent_link,
    _validate_transition,
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


class SQLiteAgentDesignArchive:
    """Durable append-only archive using stdlib sqlite3."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._memory = InMemoryAgentDesignArchive()
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self._db_path)
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

    def _load_from_db(self) -> None:
        if self._memory._history:
            return
        conn = self._connect()
        rows = conn.execute(
            "SELECT payload_json FROM adas_candidate_snapshots ORDER BY candidate_id, sequence"
        ).fetchall()
        for row in rows:
            record = AgentCandidateRecord.model_validate_json(row["payload_json"])
            self._memory._snapshots[record.snapshot_id] = record
            self._memory._history.setdefault(record.candidate_id, []).append(record)
        bundle_rows = conn.execute("SELECT payload_json FROM adas_evidence_bundles ORDER BY bundle_id").fetchall()
        for row in bundle_rows:
            bundle = AgentCandidateEvidenceBundle.model_validate_json(row["payload_json"])
            self._bundles_put_memory(bundle)

    def _bundles_put_memory(self, bundle: AgentCandidateEvidenceBundle) -> None:
        self._memory._bundles[bundle.bundle_id] = bundle

    def append_candidate_snapshot(
        self,
        record: AgentCandidateRecord,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord:
        self._load_from_db()
        _assert_scope(record, tenant_id=tenant_id, workspace_id=workspace_id)
        frozen = _freeze_record(record)
        prior_list = self._memory._history.get(frozen.candidate_id, [])
        prior = prior_list[-1] if prior_list else None
        if frozen.snapshot_id in self._memory._snapshots:
            existing = self._memory._snapshots[frozen.snapshot_id]
            if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                return existing
            raise AgentDesignArchiveError("conflicting snapshot identity")
        _validate_parent_link(self._memory._history, frozen)
        _validate_transition(prior, frozen)
        conn = self._connect()
        try:
            with conn:
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
        except sqlite3.IntegrityError as exc:
            raise AgentDesignArchiveError("conflicting snapshot identity") from exc
        self._memory._snapshots[frozen.snapshot_id] = frozen
        self._memory._history.setdefault(frozen.candidate_id, []).append(frozen)
        return frozen

    def get_latest_candidate_snapshot(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        self._load_from_db()
        return self._memory.get_latest_candidate_snapshot(candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)

    def get_candidate_history(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        self._load_from_db()
        return self._memory.get_candidate_history(candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)

    def list_candidates_for_objective(
        self,
        objective_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
        status: AgentCandidateStatus | None = None,
    ) -> list[AgentCandidateRecord]:
        self._load_from_db()
        return self._memory.list_candidates_for_objective(
            objective_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            status=status,
        )

    def get_child_candidates(
        self,
        parent_candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[AgentCandidateRecord]:
        self._load_from_db()
        return self._memory.get_child_candidates(
            parent_candidate_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    def get_parent_candidate(
        self,
        candidate_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateRecord | None:
        self._load_from_db()
        return self._memory.get_parent_candidate(candidate_id, tenant_id=tenant_id, workspace_id=workspace_id)

    def append_sealed_evidence_bundle(
        self,
        bundle: AgentCandidateEvidenceBundle,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle:
        self._load_from_db()
        if bundle.tenant_id != tenant_id or bundle.workspace_id != workspace_id:
            raise AgentDesignArchiveScopeError("bundle scope mismatch")
        if not bundle.sealed:
            raise AgentDesignArchiveError("bundle must be sealed before append")
        digest = bundle.bundle_digest or compute_evidence_bundle_digest(bundle)
        payload = bundle.model_dump(mode="python")
        payload["bundle_digest"] = digest
        frozen = AgentCandidateEvidenceBundle.model_validate(payload)
        verify_evidence_bundle_digest(frozen)
        if frozen.bundle_id in self._memory._bundles:
            existing = self._memory._bundles[frozen.bundle_id]
            if existing.model_dump(mode="json") == frozen.model_dump(mode="json"):
                return existing
            raise AgentDesignArchiveError("conflicting evidence bundle identity")
        conn = self._connect()
        try:
            with conn:
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
        except sqlite3.IntegrityError as exc:
            raise AgentDesignArchiveError("conflicting evidence bundle identity") from exc
        self._bundles_put_memory(frozen)
        return frozen

    def get_sealed_evidence_bundle(
        self,
        bundle_id: str,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> AgentCandidateEvidenceBundle | None:
        self._load_from_db()
        return self._memory.get_sealed_evidence_bundle(bundle_id, tenant_id=tenant_id, workspace_id=workspace_id)
