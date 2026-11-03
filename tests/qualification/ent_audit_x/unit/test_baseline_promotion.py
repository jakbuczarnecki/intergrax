from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.baseline import (
    BaselinePhase,
    EnterpriseAuditBaseline,
    baseline_path,
    candidate_baseline_path,
    promote_accepted_baseline,
    resolve_baseline_phase,
    save_candidate_baseline,
    verify_operator_decision_record,
)
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_baseline_phase_no_accepted_without_operator(tmp_path: Path) -> None:
    assert resolve_baseline_phase(tmp_path) is BaselinePhase.NO_ACCEPTED


def test_save_accepted_without_od_record_fails(tmp_path: Path) -> None:
    save_candidate_baseline(
        candidate_baseline_path(tmp_path),
        EnterpriseAuditBaseline(accepted_baseline_sha="a" * 40, findings={}),
    )
    with pytest.raises(ValueError, match="OD-ENT-01"):
        promote_accepted_baseline(tmp_path, od_record_path=tmp_path / "missing.json")


def test_promote_without_record_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        verify_operator_decision_record(tmp_path / "nope.json")


def test_promote_rejects_agent_only_record(tmp_path: Path) -> None:
    record = tmp_path / "od.json"
    record.write_text(
        json.dumps(
            {
                "decision_id": "OD-ENT-01",
                "record_origin": "agent",
                "operator_origin": False,
                "b6_acceptance_sha": "b" * 40,
                "evidence_publication_sha": "c" * 40,
                "test_execution_sha": "d" * 40,
                "accepted_baseline_sha": "e" * 40,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="agent-origin"):
        verify_operator_decision_record(record)


def _isolated_baseline_dir(root: Path) -> Path:
    path = root / "tests" / "qualification" / "ent_audit_x" / ".pytest-baseline" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_bootstrap_never_writes_accepted(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    qual = _isolated_baseline_dir(root)
    try:
        runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
        runner.run(
            AuditMode.FREEZE,
            output_dir=tmp_path / "out",
            baseline_dir=qual,
            bootstrap_baseline=True,
        )
        assert not baseline_path(qual).is_file()
    finally:
        shutil.rmtree(qual.parent, ignore_errors=True)


def test_report_certifying_run_false_on_dry_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    report, manifest, _ = EnterpriseAuditRunner(root, dry_run=True).run(
        AuditMode.FAST,
        output_dir=tmp_path / "cert",
    )
    assert report.certifying_run is False
    assert manifest.certifying_run is False


def test_dry_run_exit_incomplete(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    report, _, exit_code = EnterpriseAuditRunner(root, dry_run=True).run(
        AuditMode.FAST,
        output_dir=tmp_path / "dry",
    )
    assert exit_code is EnterpriseAuditExitCode.INCOMPLETE
    assert report.status.value in {"INCOMPLETE", "PARTIAL"}


def test_certifying_mode_rejects_check_id_filter(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    _, _, code = EnterpriseAuditRunner(root, dry_run=True).run(
        AuditMode.FAST,
        output_dir=tmp_path / "filter",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
        certifying_run=True,
    )
    assert code is EnterpriseAuditExitCode.VIOLATION


def test_promotion_requires_operator_records_path(tmp_path: Path) -> None:
    sha = "f" * 40
    save_candidate_baseline(
        candidate_baseline_path(tmp_path),
        EnterpriseAuditBaseline(accepted_baseline_sha=sha, findings={}),
    )
    od = tmp_path / "od.json"
    od.write_text(
        json.dumps(
            {
                "decision_id": "OD-ENT-01",
                "record_origin": "operator",
                "operator_origin": True,
                "b6_acceptance_sha": "a" * 40,
                "evidence_publication_sha": "b" * 40,
                "test_execution_sha": "c" * 40,
                "accepted_baseline_sha": sha,
            }
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[4]
    with pytest.raises(ValueError, match="operator_records"):
        promote_accepted_baseline(tmp_path, od_record_path=od, repo_root=root)
