from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest


def test_full_mode_plans_all_thirteen_ten_star_checks() -> None:
    checks = load_default_manifest()
    full = checks_for_mode(checks, AuditMode.FULL)
    ten = [c for c in full if c.check_id.startswith("TEN-")]
    assert len(ten) == 13


def test_cli_fast_dry_run_exit_zero(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/run_enterprise_audit.py"
    out = tmp_path / "cli"
    completed = subprocess.run(
        [sys.executable, str(script), "--mode", "FAST", "--dry-run", "--output-dir", str(out)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
