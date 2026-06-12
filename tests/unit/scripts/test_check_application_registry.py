# © Artur Czarnecki. All rights reserved.

"""APP-OPS-4 — application registry CI script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_check_application_registry_passes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "check_application_registry.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "application registry gate: OK" in completed.stdout


def test_apps_list_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, "-m", "intergrax.cli.main", "apps", "sync"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    listed = subprocess.run(
        [sys.executable, "-m", "intergrax.cli.main", "apps", "list"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr or listed.stdout
    assert "legal" in listed.stdout


def test_envs_list_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "intergrax.cli.main",
            "envs",
            "list",
            "--app",
            "legal",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "legal-strict" in completed.stdout
