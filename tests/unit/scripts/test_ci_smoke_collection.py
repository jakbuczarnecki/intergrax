# © Artur Czarnecki. All rights reserved.

"""Verify CI smoke pytest collection includes required architecture gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.run_ci_smoke_pytest import smoke_paths

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_ARCH_REPO_GATE = (
    "tests/unit/scripts/proof/test_all_initialized_scenario_architecture.py"
    "::test_repo_gate_passes_for_all_discovered_initialized_scenarios"
)


def test_scenario_architecture_repo_gate_collected_in_ci_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *smoke_paths(),
            "-m",
            "ci_smoke",
            "--collect-only",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode in (0, 5), completed.stderr or completed.stdout
    collected = completed.stdout.replace("\\", "/")
    assert SCENARIO_ARCH_REPO_GATE in collected
