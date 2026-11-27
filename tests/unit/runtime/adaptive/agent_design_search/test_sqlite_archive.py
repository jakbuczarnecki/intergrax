# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from intergrax.runtime.adaptive.agent_design_search.archive import AgentDesignArchiveError
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
        passed = sample_record(
            sequence=1,
            status=AgentCandidateStatus.STATIC_GATE_PASSED,
            static_gate_result={"passed": True},
        )
        archive.append_candidate_snapshot(passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reopened:
        history = reopened.get_candidate_history(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 2
        latest = reopened.get_latest_candidate_snapshot(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert latest is not None and latest.status is AgentCandidateStatus.STATIC_GATE_PASSED


def test_sqlite_sealed_bundle_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-bundles.sqlite3"
    record = sample_record(status=AgentCandidateStatus.DRAFT)
    gate_failed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False},
        rejection_reason="static rejection",
    )
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
        loaded = reopened.get_sealed_evidence_bundle("bundle-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert loaded is not None
        assert loaded.bundle_digest == digest


def test_q16_import_side_effects_isolated(tmp_path: Path) -> None:
    workdir = str(tmp_path).replace("\\", "\\\\")
    script = (
        "import os, sqlite3; "
        f"os.chdir('{workdir}'); "
        "original = sqlite3.connect; "
        "sqlite3.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('sqlite import side effect')); "
        "import intergrax.runtime.adaptive.agent_design_search as adas; "
        "sqlite3.connect = original; "
        "assert adas.__all__; "
        "assert 'SQLiteAgentDesignArchive' in adas.__all__; "
        "assert list(os.listdir('.')) == []"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_two_instances_observe_committed_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-shared.sqlite3"
    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as writer:
        writer.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reader:
        assert reader.get_latest_candidate_snapshot(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_concurrent_idempotent_and_conflicting_append(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-concurrent.sqlite3"
    barrier = threading.Barrier(2)

    def idempotent_worker() -> AgentCandidateStatus | None:
        with SQLiteAgentDesignArchive(db_path) as archive:
            barrier.wait()
            record = sample_record()
            archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
            latest = archive.get_latest_candidate_snapshot(
                record.candidate_id,
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
            )
            return None if latest is None else latest.status

    t1 = threading.Thread(target=idempotent_worker)
    t2 = threading.Thread(target=idempotent_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    with SQLiteAgentDesignArchive(db_path) as archive:
        history = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 1

    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)

    results: list[str | Exception] = []

    def race(snapshot_id: str, status: AgentCandidateStatus, gate: dict[str, object], reason: str | None) -> None:
        try:
            with SQLiteAgentDesignArchive(db_path) as archive:
                snapshot = sample_record(
                    sequence=1,
                    status=status,
                    static_gate_result=gate,
                    rejection_reason=reason,
                    snapshot_id=snapshot_id,
                )
                archive.append_candidate_snapshot(snapshot, tenant_id=_TENANT, workspace_id=_WORKSPACE)
                results.append(snapshot_id)
        except AgentDesignArchiveError as exc:
            results.append(exc)

    threads = [
        threading.Thread(
            target=race,
            args=("snap-a", AgentCandidateStatus.STATIC_GATE_PASSED, {"passed": True}, None),
        ),
        threading.Thread(
            target=race,
            args=("snap-b", AgentCandidateStatus.STATIC_GATE_FAILED, {"passed": False}, "static rejection"),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with SQLiteAgentDesignArchive(db_path) as archive:
        history = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 2
        assert sum(1 for item in history if item.sequence == 1) == 1
        assert isinstance(results[0], str) or isinstance(results[1], str)
