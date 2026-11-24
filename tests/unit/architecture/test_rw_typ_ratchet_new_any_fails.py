"""NT-RW-TYP-RATCHET-ANY — one new Any above baseline fails the ratchet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_TRACKED_BASELINE = _REPO / "scripts" / "qualification" / "typ_ratchet_baseline.v1.json"


def test_rw_typ_ratchet_new_any_fails(tmp_path: Path) -> None:
    tracked = json.loads(_TRACKED_BASELINE.read_text(encoding="utf-8"))
    lowered = dict(tracked)
    metrics = dict(tracked["metrics"])
    metrics["any"] = max(0, metrics["any"] - 1)
    lowered["metrics"] = metrics
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(lowered), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "maintenance" / "check_frozen_contract_typ_ratchet.py"),
            "--baseline",
            str(baseline_path),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "TYP_RATCHET_FAIL" in proc.stdout
