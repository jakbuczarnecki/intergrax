# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P6 consolidated qualification batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.dist_scale_x.selection import DIST_SCALE_X_P6_SELECTION_BATCH

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFCUTDIR = _REPO_ROOT / "tests" / "qualification" / "dist_scale_x"


def test_dist_scale_x_p6_consolidated_batch_passes() -> None:
    for target in DIST_SCALE_X_P6_SELECTION_BATCH:
        exit_code = pytest.main(
            [
                f"--confcutdir={_CONFCUTDIR}",
                str(_REPO_ROOT / target),
                "-q",
                "--tb=short",
            ],
        )
        assert exit_code == 0, f"P6 selection target failed: {target} (exit={exit_code})"

