from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ent_launcher


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_launcher_01_bat_declares_fast_non_certifying() -> None:
    bat = _root() / "scripts/enterprise_audit/audit.bat"
    text = bat.read_text(encoding="utf-8")
    assert "--non-certifying" in text
    assert "FAST" in text


@pytest.mark.skipif(sys.platform != "win32", reason="C4-E #1 Windows BAT")
def test_launcher_01_bat_unknown_arg_nonzero() -> None:
    root = _root()
    proc = subprocess.run(
        ["cmd", "/c", str(root / "scripts/enterprise_audit/audit.bat"), "not-a-mode"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_launcher_02_sh_declares_modes_and_non_certifying_fast() -> None:
    sh = _root() / "scripts/enterprise_audit/audit.sh"
    text = sh.read_text(encoding="utf-8")
    assert "fast" in text
    assert "--non-certifying" in text


@pytest.mark.skipif(sys.platform == "win32", reason="C4-E #2 requires bash")
def test_launcher_02_sh_unknown_mode_nonzero() -> None:
    root = _root()
    proc = subprocess.run(
        ["bash", str(root / "scripts/enterprise_audit/audit.sh"), "bogus"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_launcher_03_full_dispatch_declares_evidence_bundle() -> None:
    root = _root()
    bat = (root / "scripts/enterprise_audit/audit.bat").read_text(encoding="utf-8")
    sh = (root / "scripts/enterprise_audit/audit.sh").read_text(encoding="utf-8")
    assert "FULL" in bat and "--evidence-bundle" in bat
    assert "FULL" in sh and "--evidence-bundle" in sh


def test_launcher_04_changed_since_missing_since_nonzero() -> None:
    root = _root()
    if sys.platform == "win32":
        proc = subprocess.run(
            ["cmd", "/c", str(root / "scripts/enterprise_audit/audit.bat"), "changed"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        proc = subprocess.run(
            ["bash", str(root / "scripts/enterprise_audit/audit.sh"), "changed"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode != 0


@pytest.mark.skipif(sys.platform != "win32", reason="C4-E #5 Windows BAT FREEZE")
def test_launcher_05_freeze_without_accepted_blocked() -> None:
    root = _root()
    proc = subprocess.run(
        ["cmd", "/c", str(root / "scripts/enterprise_audit/audit.bat"), "freeze"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode != 0


def test_launcher_06_bat_sh_reference_uv() -> None:
    root = _root()
    assert "uv run python" in (root / "scripts/enterprise_audit/audit.bat").read_text(encoding="utf-8")
    assert "uv run python" in (root / "scripts/enterprise_audit/audit.sh").read_text(encoding="utf-8")


def test_launcher_07_invalid_since_sha_nonzero(tmp_path: Path) -> None:
    root = _root()
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "CHANGED-SINCE",
            "--since-sha",
            "not-a-valid-sha",
            "--non-certifying",
            "--output-dir",
            str(tmp_path / "bad-since"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_launcher_08_repo_root_from_subdir(tmp_path: Path) -> None:
    root = _root()
    sub = tmp_path / "nested" / "cwd"
    sub.mkdir(parents=True)
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "FAST",
            "--dry-run",
            "--non-certifying",
            "--output-dir",
            str(tmp_path / "out-subdir"),
        ],
        cwd=sub,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in {0, 2}


def test_launcher_09_paths_with_spaces_supported(tmp_path: Path) -> None:
    root = _root()
    spaced = tmp_path / "path with spaces" / "out"
    spaced.mkdir(parents=True)
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "FAST",
            "--dry-run",
            "--non-certifying",
            "--output-dir",
            str(spaced),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in {0, 2}


def test_launcher_10_exit_code_propagation_on_violation(tmp_path: Path) -> None:
    root = _root()
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "FULL",
            "--check-id",
            "ENT-A-BND-01",
            "--output-dir",
            str(tmp_path / "blocked"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_launcher_11_unique_output_dirs(tmp_path: Path) -> None:
    root = _root()
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    out_a = tmp_path / "run-a"
    out_b = tmp_path / "run-b"
    for out in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "run",
                "--mode",
                "FAST",
                "--dry-run",
                "--non-certifying",
                "--output-dir",
                str(out),
            ],
            cwd=root,
            check=False,
            capture_output=True,
        )
    assert out_a != out_b
    assert (out_a / "enterprise_audit_report.json").is_file()
    assert (out_b / "enterprise_audit_report.json").is_file()


def test_launcher_12_failed_audit_nonzero_exit(tmp_path: Path) -> None:
    root = _root()
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "FULL",
            "--check-id",
            "ENT-A-BND-01",
            "--output-dir",
            str(tmp_path / "fail"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_launcher_13_pr_workflow_file_present() -> None:
    wf = _root() / ".github/workflows/enterprise-audit.yml"
    assert wf.is_file()
    text = wf.read_text(encoding="utf-8")
    assert "pull_request" in text


def test_launcher_14_required_check_doc_present() -> None:
    root = _root()
    quickstart = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_OPERATOR_QUICKSTART.md"
    runbook = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md"
    assert quickstart.is_file() or runbook.is_file()


def test_launcher_15_report_sha_field_on_dry_run(tmp_path: Path) -> None:
    root = _root()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    out = tmp_path / "sha-check"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--mode",
            "FAST",
            "--dry-run",
            "--non-certifying",
            "--output-dir",
            str(out),
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    report = out / "enterprise_audit_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload.get("current_sha") == head


def test_launcher_16_bat_and_sh_invalid_mode_same_nonzero_semantics() -> None:
    root = _root()
    if sys.platform == "win32":
        bat = subprocess.run(
            ["cmd", "/c", str(root / "scripts/enterprise_audit/audit.bat"), "invalid"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        assert bat.returncode != 0
    if os.environ.get("ENT_AUDIT_RUN_BASH_LAUNCHER") == "1":
        sh = subprocess.run(
            ["bash", str(root / "scripts/enterprise_audit/audit.sh"), "invalid"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        assert sh.returncode != 0
