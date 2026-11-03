from __future__ import annotations

import json
import os

import pytest
from intergrax.platform.enterprise_audit.baseline import candidate_baseline_path
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.digests import manifest_digest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def _repo_root():
    return pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]


@pytest.mark.ent_l3_heavy
def test_l3_first_candidate_freeze_85_no_accepted_baseline(tmp_path) -> None:
    """First FREEZE candidate: bootstrap only; no accepted baseline authority."""
    if os.environ.get("ENT_AUDIT_RUN_L3_FREEZE") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FREEZE=1 for certifying FREEZE 85/85 (long-running)")
    root = _repo_root()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-candidate",
        baseline_dir=baseline_dir,
        bootstrap_baseline=True,
        certifying_run=True,
    )
    assert manifest.checks_planned == 85
    assert manifest.checks_executed == 85
    assert manifest.checks_skipped == 0
    assert manifest.coverage_label == "FULL"
    assert code is EnterpriseAuditExitCode.CLEAN
    assert candidate_baseline_path(baseline_dir).is_file()
    accepted = baseline_dir / "accepted_baseline.v1.json"
    assert not accepted.is_file()


@pytest.mark.ent_l3_heavy
def test_l3_laboratory_normal_freeze_85_synthetic_accepted_fixture(tmp_path) -> None:
    """Laboratory-only: synthetic accepted fixture — not operator OD-ENT-01 acceptance."""
    if os.environ.get("ENT_AUDIT_RUN_L3_FREEZE") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FREEZE=1 for certifying FREEZE 85/85 (long-running)")
    if os.environ.get("ENT_AUDIT_LAB_FREEZE") != "1":
        pytest.skip("Laboratory FREEZE requires ENT_AUDIT_LAB_FREEZE=1 (synthetic accepted fixture)")
    root = _repo_root()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    head = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    accepted = baseline_dir / "accepted_baseline.v1.json"
    accepted.write_text(
        json.dumps(
            {
                "accepted_baseline_sha": head,
                "findings": [],
                "check_manifest_digest": manifest_digest(),
                "synthetic_laboratory_fixture": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-lab",
        baseline_dir=baseline_dir,
        certifying_run=True,
    )
    assert manifest.checks_planned == 85
    assert manifest.checks_executed == 85
    assert manifest.checks_skipped == 0
    assert code is EnterpriseAuditExitCode.CLEAN


@pytest.mark.ent_l3_heavy
def test_l3_production_normal_freeze_85_after_od_ent_01(tmp_path) -> None:
    """Production FREEZE qualification only after real OD-ENT-01 operator acceptance."""
    if os.environ.get("ENT_AUDIT_OD_ENT_01") != "1":
        pytest.skip("Production FREEZE requires OD-ENT-01; do not treat lab fixture as production")
    if os.environ.get("ENT_AUDIT_RUN_L3_FREEZE") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FREEZE=1 for certifying FREEZE 85/85 (long-running)")
    root = _repo_root()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    prod_accepted = baseline_dir / "accepted_baseline.v1.json"
    if not prod_accepted.is_file():
        pytest.skip("Production accepted baseline must be provisioned under baseline_dir after OD-ENT-01")
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=True)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-production",
        baseline_dir=baseline_dir,
        certifying_run=True,
    )
    assert manifest.checks_executed == 85
    assert code is EnterpriseAuditExitCode.CLEAN


def test_l3_plain_freeze_blocked_without_baseline(tmp_path) -> None:
    root = _repo_root()
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-blocked",
        bootstrap_baseline=False,
        certifying_run=False,
    )
    assert code is EnterpriseAuditExitCode.VIOLATION
