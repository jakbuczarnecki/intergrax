# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Parts F and I: legal holds keep every holder (NT-RW-SUP-14) and the
AUDIT_EVIDENCE producer and retention composition gate (NT-RW-SUP-15).

In-process fixtures only (in-memory and temporary SQLite stores); no live backend.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.contracts.artifact_ref import ArtifactRetentionClass
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder, InMemoryGovernanceAuditSink
from intergrax.runtime.nexus.artifacts import retention
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.models import Artifact
from intergrax.runtime.nexus.artifacts.sqlite_artifact_store import SQLiteArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import ArtifactDeletionRefused, LegalHoldRecord

pytestmark = [pytest.mark.unit]

_TENANT = "tenant-a"
_T0 = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)


def _artifact(artifact_id: str = "art-1") -> Artifact:
    data = b"payload"
    return Artifact(
        tenant_id=_TENANT,
        artifact_id=artifact_id,
        run_id="run-1",
        step_id=None,
        kind="report",
        mime_type="application/octet-stream",
        created_at_utc=_T0,
        data=data,
        size_bytes=len(data),
    )


def _two_holders(store: InMemoryArtifactStore | SQLiteArtifactStore) -> None:
    store.put(_artifact())
    store.place_legal_hold(_TENANT, "art-1", placed_by="counsel-a", reason="case 1")
    store.place_legal_hold(_TENANT, "art-1", placed_by="counsel-a", reason="overwrite attempt")
    store.place_legal_hold(_TENANT, "art-1", placed_by="counsel-b", reason="case 2")


def _holders(store: InMemoryArtifactStore | SQLiteArtifactStore) -> dict[str, str]:
    reader: Callable[[str, str], tuple[LegalHoldRecord, ...]] | None = getattr(store, "legal_holds", None)
    assert reader is not None, "the store keeps no holder or reason per hold (F-D8-14)"
    return {record.placed_by: record.reason for record in reader(_TENANT, "art-1")}


# --------------------------------------------------------------------- NT-RW-SUP-14 (F-D8-14)


def test_sup_14a_second_hold_does_not_overwrite_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts.sqlite3"
    store = SQLiteArtifactStore(str(db_path))
    _two_holders(store)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT placed_by, reason FROM execution_artifact_legal_holds ORDER BY placed_by"
        ).fetchall()
    conn.close()
    assert rows == [("counsel-a", "case 1"), ("counsel-b", "case 2")]
    assert _holders(store) == {"counsel-a": "case 1", "counsel-b": "case 2"}
    assert store.is_under_legal_hold(_TENANT, "art-1") is True
    with pytest.raises(ArtifactDeletionRefused):
        store.delete_artifact(_TENANT, "art-1")


def test_sup_14b_second_hold_does_not_overwrite_in_memory() -> None:
    store = InMemoryArtifactStore()
    _two_holders(store)
    assert _holders(store) == {"counsel-a": "case 1", "counsel-b": "case 2"}
    assert store.is_under_legal_hold(_TENANT, "art-1") is True
    store.release_legal_hold(_TENANT, "art-1")
    assert store.is_under_legal_hold(_TENANT, "art-1") is False
    assert _holders(store) == {}


def test_sup_14c_old_table_migrated_without_loss(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE execution_artifact_legal_holds (
                tenant_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                placed_by TEXT NOT NULL,
                reason TEXT NOT NULL,
                placed_at_utc TEXT NOT NULL,
                PRIMARY KEY (tenant_id, artifact_id)
            )
            """
        )
        conn.execute(
            "INSERT INTO execution_artifact_legal_holds VALUES (?,?,?,?,?)",
            (_TENANT, "art-1", "counsel-a", "case 1", _T0.isoformat()),
        )
    conn.close()

    store = SQLiteArtifactStore(str(db_path))
    store.put(_artifact())
    assert store.is_under_legal_hold(_TENANT, "art-1") is True
    store.place_legal_hold(_TENANT, "art-1", placed_by="counsel-b", reason="case 2")
    assert _holders(store) == {"counsel-a": "case 1", "counsel-b": "case 2"}

    reopened = SQLiteArtifactStore(str(db_path))
    assert _holders(reopened) == {"counsel-a": "case 1", "counsel-b": "case 2"}
    with sqlite3.connect(db_path) as conn:
        backup_rows = conn.execute("SELECT placed_by FROM execution_artifact_legal_holds_v1_backup").fetchall()
    conn.close()
    assert backup_rows == [("counsel-a",)]


# --------------------------------------------------------------------- NT-RW-SUP-15 (EX-D8-01)


def test_sup_15a_audit_evidence_producer() -> None:
    store = InMemoryArtifactStore()
    producer = getattr(retention, "produce_audit_evidence_artifact", None)
    assert producer is not None, "no AUDIT_EVIDENCE producer exists (EX-D8-01)"
    stored = producer(
        store,
        tenant_id=_TENANT,
        run_id="run-1",
        artifact_id="evidence-1",
        evidence_kind="security.audit_export",
        data=b'{"ok": true}',
        clock=lambda: _T0,
    )
    assert stored.retention_class is ArtifactRetentionClass.AUDIT_EVIDENCE
    assert stored.expires_at_utc is None
    assert store.get(_TENANT, "evidence-1").retention_class is ArtifactRetentionClass.AUDIT_EVIDENCE
    with pytest.raises(ArtifactDeletionRefused) as excinfo:
        store.delete_artifact(_TENANT, "evidence-1")
    assert excinfo.value.reason == "audit_evidence"


def test_sup_15b_production_requires_retention_composition() -> None:
    store = InMemoryArtifactStore()
    audit = GovernanceAuditRecorder(InMemoryGovernanceAuditSink())
    registry = TaskExecutionRegistry()
    gate = getattr(retention, "require_retention_composition", None)
    assert gate is not None, "no retention composition gate exists (EX-D8-01)"

    with pytest.raises(ValueError) as excinfo:
        gate(production_mode=True, artifact_store=None, audit=None, sweep_registry=None)
    for name in ("artifact_store", "audit_recorder", "retention_sweep_registration"):
        assert name in str(excinfo.value)
    with pytest.raises(ValueError, match="retention_sweep_registration"):
        gate(production_mode=True, artifact_store=store, audit=audit, sweep_registry=registry)

    retention.register_retention_sweep(registry, retention.ArtifactRetentionSweeper(store, audit=audit))
    gate(production_mode=True, artifact_store=store, audit=audit, sweep_registry=registry)
    gate(production_mode=False, artifact_store=None, audit=None, sweep_registry=None)
