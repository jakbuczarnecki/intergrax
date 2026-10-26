# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P1 deterministic qualification batch (L1 + L2 only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x.catalog import DIST_SCALE_X_P1_SELECTION_BATCH

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dist_scale_x_p1_l1_l2_batch_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *DIST_SCALE_X_P1_SELECTION_BATCH, "-q", "--tb=short"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
