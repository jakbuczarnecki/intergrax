from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_manifest_check_ids_match_annex_c() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/check_manifest_annex_c_alignment.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
