from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_l3_qual_wave_orchestrator_exists() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_qual_wave.py"
    assert script.is_file()


def test_l3_qual_wave_dry_run_lists_stages() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_qual_wave.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ent_l1" in proc.stdout
    assert "ent_l2" in proc.stdout


def test_l3_qual_wave_fail_mid_propagates_exit_code() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_qual_wave.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--inject-fail-after", "l1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
