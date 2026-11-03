from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_mechanical_readiness_script_emits_fail_closed_report() -> None:
    root = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/enterprise_audit/mechanical_readiness_check.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    report_path = root / ".tmp/session/ent-audit-x-mechanical-readiness/mechanical_readiness_report.json"
    assert report_path.is_file(), completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["formally_certified"] is False
    ready = report["technical_ready_for_independent_b6_mechanical"]
    if ready:
        assert completed.returncode == 0
    else:
        assert completed.returncode == 1


def test_audit_sh_exists_and_executable_bit_not_required_on_windows() -> None:
    root = Path(__file__).resolve().parents[4]
    assert (root / "scripts/enterprise_audit/audit.sh").is_file()
    assert (root / "scripts/enterprise_audit/audit.bat").is_file()
