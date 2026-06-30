# © Artur Czarnecki. All rights reserved.

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@pytest.mark.unit
def test_agents_no_pseudo_shared_packages() -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "check_agents_no_pseudo_shared_packages.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
