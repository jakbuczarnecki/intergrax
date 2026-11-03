from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_exit_codes_fast_dry_run(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "FAST",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "fast"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
