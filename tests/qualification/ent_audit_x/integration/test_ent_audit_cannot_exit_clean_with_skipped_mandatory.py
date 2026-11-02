from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_full_mode_exit_not_clean_when_check_skipped(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    checks = load_default_manifest()
    only_one = frozenset({checks[0].check_id})
    runner = EnterpriseAuditRunner(root, dry_run=False)
    _, _, code = runner.run(
        AuditMode.FULL,
        output_dir=tmp_path / "partial",
        check_ids_filter=only_one,
    )
    assert code is not EnterpriseAuditExitCode.CLEAN
