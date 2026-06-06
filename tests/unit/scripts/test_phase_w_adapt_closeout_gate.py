from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase_w_adapt_closeout_gate_enforce_l4_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "phase_w_adapt_closeout_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--skip-scripts", "--enforce-l4-runtime"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "runtime_l4_closed_loop_passed: True" in completed.stdout
