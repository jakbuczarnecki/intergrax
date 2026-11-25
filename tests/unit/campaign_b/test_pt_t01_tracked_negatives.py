# © Artur Czarnecki. All rights reserved.

"""PT-T01 tracked negative probes: run ACTIVE matrix nodes (NT-01..NT-16)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.unit.campaign_b._pt_t01_matrix import ACTIVE_TRACKED_PYTEST_ARGS

_REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("pytest_node", ACTIVE_TRACKED_PYTEST_ARGS)
def test_pt_t01_tracked_negative_probe_passes(pytest_node: str) -> None:
    rel = pytest_node.split("::", 1)[0]
    assert (_REPO_ROOT / rel).is_file(), rel
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        pytest_node,
        "-q",
        "--tb=short",
        "-o",
        "console_output_style=classic",
    ]
    completed = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"{pytest_node} exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-4000:]}\n"
        f"stderr:\n{completed.stderr[-2000:]}"
    )
