from __future__ import annotations

import json
from pathlib import Path

from intergrax.platform.enterprise_audit.validators import validate_report_semantic_consistency


def test_adv31_mutated_status_not_clean(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "adv31")
    report_path = tmp_path / "adv31" / "enterprise_audit_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["status"] = "PASS"
    payload["findings"] = [
        {
            "finding_id": "INJECT:BLOCKER",
            "check_id": "ENT-A-BND-01",
            "severity": "BLOCKER",
            "category": "architecture",
            "message": "injected",
        }
    ]
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    errors = validate_report_semantic_consistency(report_path)
    assert errors


def test_report_status_mutation_detected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "tamper")
    report_path = tmp_path / "tamper" / "enterprise_audit_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["exit_code"] = 0
    payload["status"] = "PASS"
    payload["findings"] = [
        {
            "finding_id": "TAMPER:BLOCKER",
            "check_id": "ENT-A-BND-01",
            "severity": "BLOCKER",
            "category": "architecture",
            "message": "tampered",
        }
    ]
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert validate_report_semantic_consistency(report_path)


def test_clean_report_passes_semantic_verify(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "clean")
    report_path = tmp_path / "clean" / "enterprise_audit_report.json"
    assert not validate_report_semantic_consistency(report_path)
