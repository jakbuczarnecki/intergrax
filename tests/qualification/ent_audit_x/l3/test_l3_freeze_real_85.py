from __future__ import annotations

import os

import pytest
from intergrax.platform.enterprise_audit.baseline import candidate_baseline_path
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.digests import manifest_digest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


@pytest.mark.ent_l3_heavy
def test_l3_first_candidate_freeze_85(tmp_path) -> None:
    if os.environ.get("ENT_AUDIT_RUN_L3_FREEZE") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FREEZE=1 for certifying FREEZE 85/85 (long-running)")
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]
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


@pytest.mark.ent_l3_heavy
def test_l3_normal_freeze_85_with_accepted_fixture(tmp_path) -> None:
    if os.environ.get("ENT_AUDIT_RUN_L3_FREEZE") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FREEZE=1 for certifying FREEZE 85/85 (long-running)")
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    accepted = baseline_dir / "accepted_baseline.v1.json"
    head = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    accepted.write_text(
        __import__("json").dumps(
            {
                "accepted_baseline_sha": head,
                "findings": [],
                "check_manifest_digest": manifest_digest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-normal",
        baseline_dir=baseline_dir,
        certifying_run=True,
    )
    assert manifest.checks_planned == 85
    assert manifest.checks_executed == 85
    assert manifest.checks_skipped == 0
    assert code is EnterpriseAuditExitCode.CLEAN


def test_l3_plain_freeze_blocked_without_baseline(tmp_path) -> None:
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-blocked",
        bootstrap_baseline=False,
        certifying_run=False,
    )
    assert code is EnterpriseAuditExitCode.VIOLATION
