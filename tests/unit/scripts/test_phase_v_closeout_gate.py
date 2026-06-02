from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase_v_closeout_gate_skip_scripts_enforces_l3() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "phase_v_closeout_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--skip-scripts", "--enforce"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
