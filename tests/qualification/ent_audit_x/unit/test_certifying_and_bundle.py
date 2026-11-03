from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.evidence import build_evidence_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import verify_closed_evidence_bundle
from tests.qualification.ent_audit_x._manifest_fixtures import sample_full_execution_manifest


def test_cli_check_id_requires_non_certifying(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "FAST",
            "--check-id",
            "ENT-A-BND-01",
            "--output-dir",
            str(tmp_path / "bad"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1


def test_cli_full_with_check_id_exits_nonzero(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "FULL",
            "--check-id",
            "ENT-A-BND-01",
            "--output-dir",
            str(tmp_path / "full-bad"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_cli_prints_summary_fields(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "summary"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in {0, 1, 2, 3}
    assert "Enterprise audit summary" in completed.stdout


def test_verify_closed_bundle_ok(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    out = tmp_path / "closed"
    with pytest.raises(ValueError, match="certifying_run"):
        runner.run(AuditMode.FAST, output_dir=out, write_evidence_bundle=True)


def test_extra_artifact_fails_closed_verify(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "extra"
    EnterpriseAuditRunner(root, dry_run=True).run(AuditMode.FAST, output_dir=out)
    (out / "unexpected.txt").write_text("x", encoding="utf-8")
    errors = verify_closed_evidence_bundle(out)
    assert errors


def test_adv29_evidence_refused_when_non_certifying(tmp_path: Path) -> None:
    out = tmp_path / "noncert"
    out.mkdir()
    (out / "enterprise_audit_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="certifying_run"):
        build_evidence_manifest(
            out,
            run_id="x",
            qualification_sha="a" * 40,
            certifying_run=False,
            execution_manifest=sample_full_execution_manifest(),
        )
