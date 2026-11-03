from __future__ import annotations

import os

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


@pytest.mark.ent_l3_heavy
def test_l3_full_real_85_of_85(tmp_path) -> None:
    if os.environ.get("ENT_AUDIT_RUN_L3_FULL") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FULL=1 for certifying FULL 85/85 (long-running)")
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, manifest, code = runner.run(
        AuditMode.FULL,
        output_dir=tmp_path / "l3-full-real-85",
        certifying_run=True,
    )
    assert manifest.checks_planned == 85
    assert manifest.checks_executed == 85
    assert manifest.checks_skipped == 0
    assert manifest.coverage_label == "FULL"
    assert manifest.certifying_run is True
    assert code is EnterpriseAuditExitCode.CLEAN


def test_l3_full_real_85_fail_on_check_filter(tmp_path) -> None:
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[4]
    from intergrax.platform.enterprise_audit.contracts import EnterpriseAuditExitCode

    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FULL,
        output_dir=tmp_path / "l3-full-filter-blocked",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
        certifying_run=True,
    )
    assert code is EnterpriseAuditExitCode.VIOLATION
