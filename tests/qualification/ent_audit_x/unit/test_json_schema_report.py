from __future__ import annotations

import json
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import validate_report_json


def test_json_report_validates_against_schema(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "out")
    report_path = tmp_path / "out" / "enterprise_audit_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("status", "exit_code", "audit_mode", "execution_manifest"):
        assert key in payload
    assert not validate_report_json(report_path)
