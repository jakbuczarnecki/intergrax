# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_check_agent_acp_close_ci_gate_passes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "gates" / "check_agent_acp_close_ci.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ACP-CLOSE CI gate: OK" in completed.stdout
