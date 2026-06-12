# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-6 — environment diff CI script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_check_application_environment_diff_passes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "check_application_environment_diff.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "application environment diff gate: OK" in completed.stdout


def test_doctor_diff_app_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "intergrax.cli.main",
            "doctor",
            "diff-app",
            "--app",
            "legal",
            "--left",
            "0.1.0",
            "--right",
            "0.2.0",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "risk:" in completed.stdout
