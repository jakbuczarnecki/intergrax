from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import scripts.enterprise_audit.run_qual_wave as qw


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_pre_qual_status_not_fully_qualified_when_l3_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qw, "git_head", lambda _r: "abc123")
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "_run_pytest", lambda *a, **k: 0)
    report = qw.run_pre_qual_stages(_root(), run_l3_cli=False, l3_output=tmp_path / "l3")
    assert report.status == "PRE_QUAL_PASS"
    skipped = [s for s in report.stages if s.name == "l3_cli_full" and s.skipped]
    assert skipped
    assert not report.certifying


def test_certifying_fails_when_l3_cli_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qw, "git_head", lambda _r: "abc123")
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "_run_pytest", lambda *a, **k: 0)
    monkeypatch.setattr(qw, "_run_cli_full", lambda *a, **k: (1, "cli failed"))
    report = qw.run_pre_qual_stages(_root(), run_l3_cli=True, l3_output=tmp_path / "l3")
    assert report.status != "PASS"
    assert "L3 CLI" in report.error


def test_certifying_fails_on_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qw, "git_head", lambda _r: "abc123")
    monkeypatch.setattr(qw, "git_clean", lambda _r: False)
    report = qw.run_pre_qual_stages(_root(), run_l3_cli=True, l3_output=tmp_path / "l3")
    assert report.error == "working tree not clean"


def test_acceptance_tests_not_collected_as_ent_l2() -> None:
    root = _root()
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/qualification/ent_audit_x/acceptance",
            "-m",
            "ent_l2",
            "--collect-only",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "no tests collected" in proc.stdout.lower() or proc.returncode == 5


def test_orchestrator_pre_qual_writes_report(tmp_path: Path) -> None:
    import os

    if os.environ.get("ENT_AUDIT_RUN_PREQUAL") != "1":
        pytest.skip("Full pre-qual orchestrator run requires ENT_AUDIT_RUN_PREQUAL=1")
    root = _root()
    log_dir = tmp_path / "log"
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/enterprise_audit/run_qual_wave.py"),
            "--profile",
            "pre-qual",
            "--log-dir",
            str(log_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=7200,
    )
    report_path = log_dir / "qual_wave_report.json"
    assert report_path.is_file(), proc.stderr[-2000:]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if proc.returncode == 0:
        assert payload["status"] == "PRE_QUAL_PASS"
        assert payload["profile"] == "pre-qual"
        l3 = next(s for s in payload["stages"] if s["name"] == "l3_cli_full")
        assert l3["skipped"] is True
    else:
        assert payload["status"] != "PASS"


def test_orchestrator_certifying_sha_pin_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setenv("TEST_EXECUTION_SHA", "0" * 40)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/enterprise_audit/run_qual_wave.py"),
            "--profile",
            "certifying",
            "--log-dir",
            str(tmp_path / "log"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert head in proc.stderr or "TEST_EXECUTION_SHA" in proc.stderr
