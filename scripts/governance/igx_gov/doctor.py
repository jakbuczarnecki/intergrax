# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def run_doctor() -> int:
    exit_code = 0
    readiness = subprocess.run(
        [sys.executable, str(_REPO / "scripts/dev/verify_push_gate_readiness.py")],
        cwd=_REPO,
        check=False,
    )
    if readiness.returncode != 0:
        exit_code = readiness.returncode

    for script in (
        "validate_policy_registry.py",
        "verify_agent_control_plane.py",
        "verify_control_plane_consistency.py",
        "verify_agent_skills.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(_REPO / "scripts/governance" / script)],
            cwd=_REPO,
            check=False,
        )
        if completed.returncode != 0:
            exit_code = completed.returncode

    manifest = _REPO / "governance" / "GOVERNANCE-MANIFEST.yaml"
    if not manifest.is_file():
        print("[ERROR] doctor: missing GOVERNANCE-MANIFEST.yaml", file=sys.stderr)
        exit_code = 1

    if exit_code == 0:
        print("[INFO] igx_gov doctor: OK")
    else:
        print("[ERROR] igx_gov doctor: failed", file=sys.stderr)
    return exit_code
