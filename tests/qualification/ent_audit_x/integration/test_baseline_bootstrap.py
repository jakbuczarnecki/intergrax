from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_freeze_blocks_no_baseline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(AuditMode.FREEZE, output_dir=tmp_path / "freeze")
    assert code is EnterpriseAuditExitCode.VIOLATION


def test_bootstrap_baseline_flag_allowed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "bootstrap",
        bootstrap_baseline=True,
    )
    assert code in {EnterpriseAuditExitCode.CLEAN, EnterpriseAuditExitCode.INCOMPLETE}
