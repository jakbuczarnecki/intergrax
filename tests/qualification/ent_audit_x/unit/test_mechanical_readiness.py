from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_mechanical_readiness_script_passes() -> None:
    root = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/enterprise_audit/mechanical_readiness_check.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_audit_sh_exists_and_executable_bit_not_required_on_windows() -> None:
    root = Path(__file__).resolve().parents[4]
    assert (root / "scripts/enterprise_audit/audit.sh").is_file()
    assert (root / "scripts/enterprise_audit/audit.bat").is_file()
