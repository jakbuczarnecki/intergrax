# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from intergrax.runtime.adaptive.agent_design_search.archive import (
    AgentDesignArchiveError,
    AgentDesignArchiveScopeError,
)
from intergrax.runtime.adaptive.agent_design_search.contracts import (
    AdasJsonValue,
    AgentCandidateEvidenceBundle,
    AgentCandidateStatus,
    W1TerminalEvidenceOutcome,
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
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
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
    pure_dir = tmp_path / "pure"
    guard_dir = tmp_path / "guard"
    pure_dir.mkdir()
    guard_dir.mkdir()
    workdir = str(pure_dir).replace("\\", "\\\\")
    guard_workdir = str(guard_dir).replace("\\", "\\\\")
    pure_import_script = f"""
import os
import socket
import sqlite3
import subprocess
import sys
import urllib.request

os.chdir({workdir!r})
side_effects = {{}}
profiled_calls = []
before = set(sys.modules)

def _deny_sqlite(*_a, **_k):
    side_effects['sqlite_connect'] = side_effects.get('sqlite_connect', 0) + 1
    raise AssertionError('sqlite import side effect')

def _deny_subprocess_run(*_a, **_k):
    side_effects['subprocess'] = side_effects.get('subprocess', 0) + 1
    raise AssertionError('subprocess import side effect')

class _GuardPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        side_effects['subprocess'] = side_effects.get('subprocess', 0) + 1
        raise AssertionError('subprocess import side effect')

def _deny_network(*_a, **_k):
    side_effects['network'] = side_effects.get('network', 0) + 1
    raise AssertionError('network import side effect')

sqlite3.connect = _deny_sqlite
subprocess.Popen = _GuardPopen
subprocess.run = _deny_subprocess_run
socket.create_connection = _deny_network
urllib.request.urlopen = _deny_network

def _profiler(frame, event, arg):
    if event == 'call':
        mod = frame.f_globals.get('__name__', '')
        fn = frame.f_code.co_name
        if (
            mod.startswith('intergrax.providers')
            or (
                mod == 'intergrax.runtime.task.unified_task_runner'
                and fn in {{'run_task', 'run_runtime_request', '__init__'}}
            )
            or (mod == 'intergrax.runtime.adaptive.profile_promotion' and fn == 'evaluate_profile_promotion')
        ):
            profiled_calls.append((mod, fn))

sys.setprofile(_profiler)
try:
    import intergrax.runtime.adaptive.agent_design_search as adas  # noqa: E402
finally:
    sys.setprofile(None)

assert adas.__all__
assert 'SQLiteAgentDesignArchive' in adas.__all__
assert side_effects == {{}}, side_effects
assert profiled_calls == [], profiled_calls
after = set(sys.modules)
delta = after - before
assert 'intergrax.providers' not in after
assert 'intergrax.runtime.promotion' not in after
assert not any(name.startswith('intergrax.providers') for name in delta)
assert list(os.listdir('.')) == []
"""
    import os
    from pathlib import Path as PathLib

    repo_root = str(PathLib(__file__).resolve().parents[5])
    env = {**os.environ, "PYTHONPATH": repo_root}
    pure_completed = subprocess.run(
        [sys.executable, "-c", pure_import_script], check=False, capture_output=True, text=True, env=env
    )
    assert pure_completed.returncode == 0, pure_completed.stderr or pure_completed.stdout

    guard_script = f"""
import os
import sys

os.chdir({guard_workdir!r})
side_effects = {{}}

def _deny_dispatch(*_a, **_k):
    side_effects['dispatch'] = side_effects.get('dispatch', 0) + 1
    raise AssertionError('dispatch import side effect')

def _deny_promotion(*_a, **_k):
    side_effects['promotion'] = side_effects.get('promotion', 0) + 1
    raise AssertionError('promotion import side effect')

import intergrax.runtime.task.unified_task_runner as utr  # noqa: E402
import intergrax.runtime.adaptive.profile_promotion as pp  # noqa: E402

utr.UnifiedTaskRunner.run_task = _deny_dispatch
utr.UnifiedTaskRunner.run_runtime_request = _deny_dispatch
pp.evaluate_profile_promotion = _deny_promotion

import intergrax.runtime.adaptive.agent_design_search as adas  # noqa: E402

assert adas.__all__
assert side_effects == {{}}, side_effects

calibration_caught = False
try:
    pp.evaluate_profile_promotion()
except AssertionError:
    calibration_caught = True
assert calibration_caught, 'guard calibration failed'
assert side_effects.get('promotion') == 1, side_effects
"""
    guard_completed = subprocess.run(
        [sys.executable, "-c", guard_script], check=False, capture_output=True, text=True, env=env
    )
    assert guard_completed.returncode == 0, guard_completed.stderr or guard_completed.stdout


def test_two_instances_observe_committed_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-shared.sqlite3"
    record = sample_record()
    writer = SQLiteAgentDesignArchive(db_path)
    reader = SQLiteAgentDesignArchive(db_path)
    writer._connect()
    reader._connect()
    writer.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert reader.get_latest_candidate_snapshot(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    passed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_PASSED,
        static_gate_result={"passed": True},
    )
    writer.append_candidate_snapshot(passed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    latest = reader.get_latest_candidate_snapshot(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert latest is not None and latest.sequence == 1
    writer.close()
    reader.close()


def test_concurrent_idempotent_and_conflicting_append(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-concurrent.sqlite3"
    barrier = threading.Barrier(2)
    duplicate_outcomes: list[str] = []
    errors: list[str] = []

    def idempotent_worker(slot: int) -> None:
        try:
            with SQLiteAgentDesignArchive(db_path) as archive:
                barrier.wait(timeout=5)
                record = sample_record()
                archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
                duplicate_outcomes.append(f"ok-{slot}")
        except Exception as exc:  # pragma: no cover - defensive witness
            errors.append(str(exc))

    workers = [threading.Thread(target=idempotent_worker, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive(), "concurrent duplicate worker did not finish"
    assert len(duplicate_outcomes) == 2
    with SQLiteAgentDesignArchive(db_path) as archive:
        history = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 1

    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)

    race_barrier = threading.Barrier(2)
    results: list[str] = []

    def race(
        snapshot_id: str,
        status: AgentCandidateStatus,
        gate: dict[str, AdasJsonValue],
        reason: str | None,
    ) -> None:
        try:
            with SQLiteAgentDesignArchive(db_path) as archive:
                race_barrier.wait(timeout=5)
                snapshot = sample_record(
                    sequence=1,
                    status=status,
                    static_gate_result=gate,
                    rejection_reason=reason,
                    snapshot_id=snapshot_id,
                )
                archive.append_candidate_snapshot(snapshot, tenant_id=_TENANT, workspace_id=_WORKSPACE)
                results.append(f"ok:{snapshot_id}")
        except AgentDesignArchiveError:
            results.append(f"err:{snapshot_id}")

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
        thread.join(timeout=5)
        assert not thread.is_alive(), "concurrent conflict worker did not finish"
    assert len([item for item in results if item.startswith("ok:")]) == 1
    assert len([item for item in results if item.startswith("err:")]) == 1
    with SQLiteAgentDesignArchive(db_path) as archive:
        history = archive.get_candidate_history("cand-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 2
        assert sum(1 for item in history if item.sequence == 1) == 1


def test_q14_reopen_preserves_idempotency(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-idempotent-reopen.sqlite3"
    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reopened:
        reopened.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        history = reopened.get_candidate_history(record.candidate_id, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(history) == 1


def test_q15_tampered_persisted_bundle_digest_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-tamper.sqlite3"
    gate_failed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False},
        rejection_reason="static",
    )
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-tamper",
        candidate_id=gate_failed.candidate_id,
        objective_id=gate_failed.objective_id,
        tenant_id=gate_failed.tenant_id,
        workspace_id=gate_failed.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest="a" * 64,
        candidate_configuration_digest=gate_failed.candidate_configuration_digest,
        candidate_record=gate_failed,
        static_gate_result={"passed": False},
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(sample_record(), tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_candidate_snapshot(gate_failed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    import sqlite3

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT payload_json FROM adas_evidence_bundles WHERE bundle_id = ?",
        ("bundle-tamper",),
    ).fetchone()
    assert row is not None
    payload = row[0].replace(digest, "f" * 64)
    conn.execute(
        "UPDATE adas_evidence_bundles SET payload_json = ? WHERE bundle_id = ?",
        (payload, "bundle-tamper"),
    )
    conn.commit()
    conn.close()
    with SQLiteAgentDesignArchive(db_path) as archive:
        with pytest.raises(AgentDesignArchiveError):
            archive.get_sealed_evidence_bundle("bundle-tamper", tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_q15_missing_persisted_bundle_digest_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-missing-digest.sqlite3"
    gate_failed = sample_record(
        sequence=1,
        status=AgentCandidateStatus.STATIC_GATE_FAILED,
        static_gate_result={"passed": False},
        rejection_reason="static",
    )
    bundle = AgentCandidateEvidenceBundle(
        bundle_id="bundle-missing",
        candidate_id=gate_failed.candidate_id,
        objective_id=gate_failed.objective_id,
        tenant_id=gate_failed.tenant_id,
        workspace_id=gate_failed.workspace_id,
        created_at=datetime.now(UTC),
        created_by="operator",
        baseline_configuration_digest="a" * 64,
        candidate_configuration_digest=gate_failed.candidate_configuration_digest,
        candidate_record=gate_failed,
        static_gate_result={"passed": False},
        terminal_evidence_outcome=W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION,
        assembled_at=datetime.now(UTC),
        sealed=True,
    )
    digest = compute_evidence_bundle_digest(bundle)
    sealed = bundle.model_copy(update={"bundle_digest": digest})
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(sample_record(), tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_candidate_snapshot(gate_failed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_sealed_evidence_bundle(sealed, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    import sqlite3

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT payload_json FROM adas_evidence_bundles WHERE bundle_id = ?",
        ("bundle-missing",),
    ).fetchone()
    assert row is not None
    import json as _json

    payload_obj = _json.loads(row[0])
    payload_obj.pop("bundle_digest", None)
    payload = _json.dumps(payload_obj)
    conn.execute(
        "UPDATE adas_evidence_bundles SET payload_json = ? WHERE bundle_id = ?",
        (payload, "bundle-missing"),
    )
    conn.commit()
    conn.close()
    with SQLiteAgentDesignArchive(db_path) as archive:
        with pytest.raises(AgentDesignArchiveError, match="missing digest|integrity violation"):
            archive.get_sealed_evidence_bundle("bundle-missing", tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_q15_failed_append_leaves_rows_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-rollback.sqlite3"
    record = sample_record()
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    bad = sample_record(
        sequence=1,
        status=AgentCandidateStatus.PROMOTED,
        static_gate_result={"passed": True},
    )
    with SQLiteAgentDesignArchive(db_path) as archive:
        conn = archive._connect()
        before = conn.execute("SELECT COUNT(*) FROM adas_candidate_snapshots").fetchone()[0]
        with pytest.raises(AgentDesignArchiveError):
            archive.append_candidate_snapshot(bad, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        after = conn.execute("SELECT COUNT(*) FROM adas_candidate_snapshots").fetchone()[0]
        assert before == after == 1
        archive.append_candidate_snapshot(
            sample_record(
                sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}
            ),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )


def test_q15_connection_reuse_after_rejected_transaction(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-retry.sqlite3"
    archive = SQLiteAgentDesignArchive(db_path)
    archive._connect()
    record = sample_record()
    archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with pytest.raises(AgentDesignArchiveError):
        archive.append_candidate_snapshot(
            sample_record(sequence=2, status=AgentCandidateStatus.EVALUATED),
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
        )
    archive.append_candidate_snapshot(
        sample_record(sequence=1, status=AgentCandidateStatus.STATIC_GATE_PASSED, static_gate_result={"passed": True}),
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    archive.close()


def test_sqlite_foreign_snapshot_collision_rejected_before_deserialize(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-scope.sqlite3"
    record = sample_record(snapshot_id="snap-shared")
    foreign = sample_record(snapshot_id="snap-shared", candidate_id="foreign-cand").model_copy(
        update={
            "tenant_id": "tenant-b",
            "envelope": sample_record().envelope.model_copy(update={"tenant_id": "tenant-b"}),
        }
    )
    with SQLiteAgentDesignArchive(db_path) as foreign_archive:
        foreign_archive.append_candidate_snapshot(foreign, tenant_id="tenant-b", workspace_id=_WORKSPACE)
    archive = SQLiteAgentDesignArchive(db_path)
    with pytest.raises(AgentDesignArchiveScopeError):
        archive.append_candidate_snapshot(record, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    archive.close()


def test_sqlite_parent_child_lineage_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "adas-lineage.sqlite3"
    parent = sample_record(candidate_id="parent-1", sequence=0)
    child = sample_record(candidate_id="child-1", sequence=0).model_copy(
        update={"parent_candidate_id": "parent-1", "lineage_id": "line-child"},
    )
    with SQLiteAgentDesignArchive(db_path) as archive:
        archive.append_candidate_snapshot(parent, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        archive.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
    with SQLiteAgentDesignArchive(db_path) as reopened:
        got_parent = reopened.get_parent_candidate("child-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert got_parent is not None and got_parent.candidate_id == "parent-1"
        kids = reopened.get_child_candidates("parent-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert [k.candidate_id for k in kids] == ["child-1"]
        parent_hist = reopened.get_candidate_history("parent-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        child_hist = reopened.get_candidate_history("child-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(parent_hist) == 1 and len(child_hist) == 1
        reopened.append_candidate_snapshot(parent, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        reopened.append_candidate_snapshot(child, tenant_id=_TENANT, workspace_id=_WORKSPACE)
        assert len(reopened.get_candidate_history("parent-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1
        assert len(reopened.get_candidate_history("child-1", tenant_id=_TENANT, workspace_id=_WORKSPACE)) == 1
