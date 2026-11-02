from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import unmapped_evol_changed_paths
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_ten_manifest_count_is_thirteen() -> None:
    checks = load_default_manifest()
    ten = [c for c in checks if c.check_id.startswith("TEN-")]
    assert len(ten) == 13


def test_vbn_manifest_count_is_nine() -> None:
    checks = load_default_manifest()
    vbn = [c for c in checks if c.check_id.startswith("VBN-")]
    assert len(vbn) == 9


def test_ten_checks_have_executable_entrypoints() -> None:
    checks = load_default_manifest()
    for check in checks:
        if not check.check_id.startswith("TEN-"):
            continue
        assert check.entrypoint_kind != "internal_pending", check.check_id


def test_vbn_checks_have_executable_entrypoints() -> None:
    checks = load_default_manifest()
    for check in checks:
        if not check.check_id.startswith("VBN-"):
            continue
        assert check.entrypoint_kind != "internal_pending", check.check_id


def test_changed_since_fail_closed_unmapped_evol_path() -> None:
    checks = load_default_manifest()
    fake = frozenset({"tests/qualification/evol_x/unmapped/fake_gate.py"})
    unmapped = unmapped_evol_changed_paths(checks, fake)
    assert "tests/qualification/evol_x/unmapped/fake_gate.py" in unmapped


def test_vbn_domain_report_not_all_na_with_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "vbn.json"
    subprocess.run(
        [sys.executable, str(root / "scripts/enterprise_audit/vbn_domain_report.py"), "--output", str(out)],
        cwd=root,
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    statuses = {d["status"] for d in payload["domains"]}
    assert statuses != {"NA_WITH_EVIDENCE"}
