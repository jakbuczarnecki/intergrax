from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_acceptance_ent_a_bnd_01_runner_subprocess_oracle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=False)
    _, outcome_manifest, code = runner.run(
        AuditMode.FAST,
        output_dir=tmp_path / "oracle-bnd01",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
        certifying_run=False,
    )
    assert outcome_manifest.checks_planned == 1
    assert outcome_manifest.checks_executed == 1
    bnd = outcome_manifest.outcomes[0]
    assert bnd.check_id == "ENT-A-BND-01"
    assert bnd.executed and bnd.exit_code == 0
    assert code in {
        EnterpriseAuditExitCode.CLEAN,
        EnterpriseAuditExitCode.INCOMPLETE,
        EnterpriseAuditExitCode.VIOLATION,
    }


def test_cli_single_check_subprocess(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts" / "enterprise_audit" / "run_enterprise_audit.py"
    out = tmp_path / "cli-one"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "FAST",
            "--non-certifying",
            "--check-id",
            "ENT-A-BND-01",
            "--output-dir",
            str(out),
        ],
        cwd=root,
        check=False,
    )
    assert completed.returncode in {0, 1, 2, 3}
    assert (out / "enterprise_audit_report.json").is_file()


def test_manifest_frz_inv_checks_wired() -> None:
    checks = load_default_manifest()
    assert all(c.entrypoint_kind != "internal_pending" for c in checks)
