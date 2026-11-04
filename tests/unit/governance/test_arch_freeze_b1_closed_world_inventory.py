# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE B1 inventory CLI and classification (T-B1-STRICT-QUALIFY)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.governance import arch_freeze_b1_closed_world_inventory as b1

pytestmark = pytest.mark.unit


def test_classify_path_strict_reachability_only_for_production_effective() -> None:
    cls, _ = b1._classify_path(
        "intergrax/runtime/foo.py",
        in_evol_inventory=True,
        reachable_from_strict=True,
        is_test=False,
    )
    assert cls == "PRODUCTION_EFFECTIVE"
    cls2, _ = b1._classify_path(
        "intergrax/runtime/bar.py",
        in_evol_inventory=True,
        reachable_from_strict=False,
        is_test=False,
    )
    assert cls2 == "PRODUCTION_ELIGIBLE_UNVERIFIED"


def test_classify_path_lab_and_test() -> None:
    assert (
        b1._classify_path("tests/unit/x.py", in_evol_inventory=False, reachable_from_strict=False, is_test=True)[0]
        == "TEST_ONLY"
    )
    assert (
        b1._classify_path(
            "agents/foo/debug_lab_x.py",
            in_evol_inventory=True,
            reachable_from_strict=True,
            is_test=False,
        )[0]
        == "LAB_ONLY"
    )


@pytest.mark.slow
def test_report_only_cli_exits_zero(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/governance/arch_freeze_b1_closed_world_inventory.py"
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(script), "--report-only", "--output-dir", str(out)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0
    summary = json.loads((out / "arch_freeze_b1_inventory_summary.json").read_text(encoding="utf-8"))
    assert summary["inventory_implementation_sha"]


@pytest.mark.slow
def test_inventory_summary_sha_fields(tmp_path: Path) -> None:
    payload = b1.run_scan(
        output_dir=tmp_path,
        audit_baseline_sha=b1.DEFAULT_AUDIT_BASELINE_SHA,
        strict_certify=False,
    )
    assert payload["execution_sha"]
    assert payload["inventory_implementation_sha"]
    assert payload["audit_baseline_sha"] == b1.DEFAULT_AUDIT_BASELINE_SHA
    assert payload["inventory_digest"]
    assert "production_eligible_unverified" in payload
