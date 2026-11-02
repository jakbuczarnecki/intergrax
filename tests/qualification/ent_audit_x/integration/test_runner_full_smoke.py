from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_runner_full_smoke_plans_eighty_five(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, manifest, _ = runner.run(AuditMode.FULL, output_dir=tmp_path / "full")
    checks = load_default_manifest()
    assert manifest.checks_planned == len(checks) == 85
