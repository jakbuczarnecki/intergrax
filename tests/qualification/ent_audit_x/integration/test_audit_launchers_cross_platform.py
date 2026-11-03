from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_bat_exists_and_lists_modes() -> None:
    root = Path(__file__).resolve().parents[4]
    bat = root / "scripts/enterprise_audit/audit.bat"
    assert bat.is_file()
    text = bat.read_text(encoding="utf-8")
    assert "FAST" in text
    assert "FULL" in text


def test_audit_sh_exists() -> None:
    root = Path(__file__).resolve().parents[4]
    sh = root / "scripts/enterprise_audit/audit.sh"
    assert sh.is_file()


def test_cli_fast_non_certifying_subprocess(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    out = tmp_path / "launcher-fast"
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
            str(out),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in {0, 2}
    report = out / "enterprise_audit_report.json"
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload.get("certifying_run") is False
