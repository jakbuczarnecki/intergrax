from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def _script() -> Path:
    return _root() / "scripts/enterprise_audit/run_qual_wave.py"


def test_l3_qual_wave_orchestrator_exists() -> None:
    assert _script().is_file()


def test_l3_qual_wave_dry_run_lists_semantic_stages() -> None:
    proc = subprocess.run(
        [sys.executable, str(_script()), "--dry-run"],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ent_semantic_gate" in proc.stdout
    assert "ent_semantic_matrix" in proc.stdout
    assert "l3_cli_full" in proc.stdout


def test_l3_qual_wave_inject_l1_failure_exits_nonzero() -> None:
    proc = subprocess.run(
        [sys.executable, str(_script()), "--inject-fail-after", "l1"],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 2


def test_l3_qual_wave_inject_l2_runs_integration_marker() -> None:
    proc = subprocess.run(
        [sys.executable, str(_script()), "--inject-fail-after", "l2"],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode in (0, 2)


def test_l3_qual_wave_pre_qual_report_skipped_l3(tmp_path: Path) -> None:
    if os.environ.get("ENT_AUDIT_RUN_PREQUAL") != "1":
        pytest.skip("Full pre-qual orchestrator run requires ENT_AUDIT_RUN_PREQUAL=1")
    log_dir = tmp_path / "qual-log"
    proc = subprocess.run(
        [
            sys.executable,
            str(_script()),
            "--profile",
            "pre-qual",
            "--log-dir",
            str(log_dir),
        ],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
        timeout=7200,
    )
    report_path = log_dir / "qual_wave_report.json"
    if not report_path.is_file():
        assert proc.returncode != 0
        return
    data = json.loads(report_path.read_text(encoding="utf-8"))
    l3 = next(s for s in data["stages"] if s["name"] == "l3_cli_full")
    assert l3["skipped"] is True
    if proc.returncode == 0:
        assert data["profile"] == "pre-qual"
        assert data["certifying"] is False


def test_l3_qual_wave_certifying_profile_declared_in_dry_run() -> None:
    proc = subprocess.run(
        [sys.executable, str(_script()), "--dry-run", "--profile", "certifying"],
        cwd=_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "certifying" in proc.stdout
