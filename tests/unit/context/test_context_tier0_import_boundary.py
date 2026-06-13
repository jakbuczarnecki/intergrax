# © Artur Czarnecki. All rights reserved.

"""CE-1.6: Tier-0 context package import boundary gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_context_tier0_import_boundary_script() -> None:
    script = REPO_ROOT / "scripts" / "check_context_tier0_import_boundary.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
