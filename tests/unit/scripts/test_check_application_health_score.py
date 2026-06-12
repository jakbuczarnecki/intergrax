# © Artur Czarnecki. All rights reserved.

"""APP-OPS-3 — health score CI script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_check_application_health_score_passes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "check_application_health_score.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "application health score gate: OK" in completed.stdout


def test_doctor_health_app_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "intergrax.cli.main",
            "doctor",
            "health-app",
            "--app",
            "legal",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "overall:" in completed.stdout
