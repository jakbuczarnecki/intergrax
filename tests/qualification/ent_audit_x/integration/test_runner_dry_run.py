from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditMode, ManifestCheck
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox


def test_sandbox_rejects_mutating_check(tmp_path: Path) -> None:
    check = ManifestCheck(
        check_id="TEST-MUT",
        entrypoint_kind="python_script",
        entrypoint="scripts/gates/check_application_production_gates.py",
        mutating=True,
    )
    sandbox = CheckExecutionSandbox(tmp_path)
    with pytest.raises(ValueError, match="mutating"):
        sandbox.run_check(check)


def test_runner_dry_run_writes_reports(tmp_path: Path) -> None:
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    repo = Path(__file__).resolve().parents[4]
    out = tmp_path / "out"
    runner = EnterpriseAuditRunner(repo, dry_run=True, max_workers=1)
    report, manifest, exit_code = runner.run(AuditMode.FAST, output_dir=out)
    assert (out / "enterprise_audit_report.json").is_file()
    assert manifest.checks_planned >= 1
    assert report.audit_mode is AuditMode.FAST
    assert exit_code.value in {0, 1, 2, 3}
