from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import (
    validate_all_artifacts_in_dir,
    validate_report_semantic_consistency,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@pytest.mark.ent_l3_heavy
def test_l3_full_real_85_cli_certifying(tmp_path: Path) -> None:
    if os.environ.get("ENT_AUDIT_RUN_L3_FULL") != "1":
        pytest.skip("Set ENT_AUDIT_RUN_L3_FULL=1 for certifying FULL 85/85 via public CLI (long-running)")
    root = _repo_root()
    head = _git_head(root)
    os.environ["TEST_EXECUTION_SHA"] = head
    out = tmp_path / "l3-cli-full-85"
    cli = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "run",
            "--mode",
            "FULL",
            "--output-dir",
            str(out),
            "--evidence-bundle",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report_path = out / "enterprise_audit_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    em = payload["execution_manifest"]
    assert payload["current_sha"] == head
    assert em["qualification_sha"] == head
    assert em["checks_planned"] == 85
    assert em["checks_executed"] == 85
    assert em["checks_skipped"] == 0
    assert em["coverage_label"] == "FULL"
    assert payload["exit_code"] == 0
    assert payload["certifying_run"] is True
    assert not payload.get("findings")
    errors = validate_all_artifacts_in_dir(out)
    errors += validate_report_semantic_consistency(report_path)
    assert not errors, errors


def test_l3_full_real_85_fail_on_check_filter(tmp_path: Path) -> None:
    root = _repo_root()
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FULL,
        output_dir=tmp_path / "l3-full-filter-blocked",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
        certifying_run=True,
    )
    assert code is EnterpriseAuditExitCode.VIOLATION


def test_l3_full_not_qualifying_without_opt_in_profile() -> None:
    """Default pytest L3 harness must not run certifying FULL without ENT_AUDIT_RUN_L3_FULL."""
    assert os.environ.get("ENT_AUDIT_RUN_L3_FULL") != "1"
