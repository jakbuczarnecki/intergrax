from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def _isolated_baseline_dir(root: Path) -> Path:
    path = root / "tests" / "qualification" / "ent_audit_x" / ".pytest-baseline" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_freeze_blocks_no_baseline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    qual = _isolated_baseline_dir(root)
    try:
        runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
        _, _, code = runner.run(
            AuditMode.FREEZE,
            output_dir=tmp_path / "freeze",
            baseline_dir=qual,
        )
        assert code is EnterpriseAuditExitCode.VIOLATION
    finally:
        shutil.rmtree(qual, ignore_errors=True)


def test_bootstrap_baseline_flag_allowed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    qual = _isolated_baseline_dir(root)
    try:
        runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
        _, _, code = runner.run(
            AuditMode.FREEZE,
            output_dir=tmp_path / "bootstrap",
            baseline_dir=qual,
            bootstrap_baseline=True,
        )
        assert code in {EnterpriseAuditExitCode.CLEAN, EnterpriseAuditExitCode.INCOMPLETE}
        assert not (qual / "ENTERPRISE_AUDIT_BASELINE.json").is_file()
    finally:
        shutil.rmtree(qual.parent, ignore_errors=True)
