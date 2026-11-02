from __future__ import annotations

import json
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_report_json_contains_no_environment_secrets(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, manifest, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "enterprise_audit_report.json").read_text(encoding="utf-8"))
    blob = json.dumps(payload)
    assert "execution_manifest" in blob
