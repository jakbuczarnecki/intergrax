# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_VERIFY = _REPO / "scripts/governance/verify_rw_plan_phase_check3.py"


def test_rw_plan_json_exists_and_schema():
    assert _RW_JSON.is_file(), "run scripts/governance/build_rw_plan.py first"
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    assert data.get("schema") == "rw_plan.v1"
    assert len(data.get("code_fix_inventory") or []) >= 35


def test_phase_check3_cli_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(_VERIFY)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
