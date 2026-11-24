"""Steady-state: frozen contract typing ratchet passes at tracked baseline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def test_rw_typ_ratchet_baseline_unchanged_passes() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "maintenance" / "check_frozen_contract_typ_ratchet.py"),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "TYP_RATCHET_PASS" in proc.stdout
