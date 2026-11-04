from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_mechanical_readiness_evidence_integrity_fail_closed_on_defective_b5() -> None:
    root = Path(__file__).resolve().parents[4]
    ladder = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md"
    text = ladder.read_text(encoding="utf-8")
    if "87203c1b6ce9786dce0523cca29b556106b6bc8d" not in text:
        pytest.skip("R7 TES pin not in ladder")
    defective = (
        root
        / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
        / "qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d"
    )
    if not defective.is_dir():
        pytest.skip("defective B5 tree missing")
    from intergrax.platform.enterprise_audit.validators import verify_b5_publication_tree

    errors = verify_b5_publication_tree(
        root,
        defective,
        expected_qualification_sha="87203c1b6ce9786dce0523cca29b556106b6bc8d",
    )
    assert errors


def test_audit_sh_exists_and_executable_bit_not_required_on_windows() -> None:
    root = Path(__file__).resolve().parents[4]
    assert (root / "scripts/enterprise_audit/audit.sh").is_file()
    assert (root / "scripts/enterprise_audit/audit.bat").is_file()
