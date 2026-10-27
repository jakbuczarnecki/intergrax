# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P5 consolidated qualification batch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x.selection import DIST_SCALE_X_P5_SELECTION_BATCH

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dist_scale_x_p5_consolidated_batch_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *DIST_SCALE_X_P5_SELECTION_BATCH, "-q", "--tb=short"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
