# © Artur Czarnecki. All rights reserved.

"""APP-CON-7 — standalone scenario matrix CI script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.tier3_scenario]


def test_check_tier3_scenario_matrix_passes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "check_tier3_scenario_matrix.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "tier3 scenario matrix gate: OK" in completed.stdout
