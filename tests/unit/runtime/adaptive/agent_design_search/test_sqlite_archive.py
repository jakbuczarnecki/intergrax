# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AgentCandidateEvidenceBundle,
    AgentCandidateStatus,
    compute_evidence_bundle_digest,
)
from intergrax.runtime.adaptive.agent_design_search.sqlite_archive import SQLiteAgentDesignArchive
from tests.unit.runtime.adaptive.agent_design_search.test_contracts import sample_record

_TENANT = "tenant-a"
_WORKSPACE = "ws-a"


def test_sqlite_persistence_and_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-lab.sqlite3"
    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        passed = sample_record(sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED)
        archive.append_candidate_snapshot(passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reopened:
        history = reopened.get_candidate_history(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 2
        latest = reopened.get_latest_candidate_snapshot(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert latest is not None and latest.status is AgentCandidateStatus.STATIC_GATE_PASSED


def test_sqlite_sealed_bundle_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-bundles.sqlite3"
    record = sample_record(status=AgentCandidateStatus.DRAFT)
    gate_failed = sample_record(sequence=1, status=AgentCandidateStatus.STATIC_GATE_FAILED)
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-1",
        candidate_id=record.candidate_id,
        objective_id=record.objective_id,
        tenant_id=record.tenant_id,
        workspace_id=record.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest="a" * 64,
        candidate_configuration_digest="b" * 64,
        candidate_record=gate_failed,
        static_gate_result={"passed": False},
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_candidate_snapshot(gate_failed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reopened:
        assert reopened.get_sealed_evidence_bundle("bundle-1", tenant_id=_TENANT, workspace_id=_WORKSPACE) is not None


def test_sqlite_no_import_side_effects() -> None:
    import intergrax.runtime.adaptive.agent_design_search as adas_package

    assert "SQLiteAgentDesignArchive" in adas_package.__all__


def test_concurrent_conflicting_append(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-concurrent.sqlite3"
    errors: list[Exception] = []

    def worker(sequence: int) -> None:
        try:
            with SQLiteAgentDesignArchive(db_path) as archive:
                record = sample_record(sequence=sequence, snapshot_id=f"snap-{sequence}")
                if sequence == 0:
                    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
                else:
                    bad = sample_record(sequence=sequence, status=AgentCandidateStatus.EVALUATED)
                    archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors
