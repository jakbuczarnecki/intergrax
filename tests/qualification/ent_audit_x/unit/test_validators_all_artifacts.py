from __future__ import annotations

import json
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import (
    validate_against_schema,
    validate_all_artifacts_in_dir,
    validate_b5_handoff_bundle,
    validate_named_artifact,
    validate_report_json,
)


def test_extra_property_fails_closed() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
    }
    errors = validate_against_schema({"a": "ok", "unexpected": 1}, schema)
    assert any("Additional properties are not allowed" in e or "unexpected" in e.lower() for e in errors)


def test_validate_all_artifacts_in_dir_on_run_output(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "out")
    errors = validate_all_artifacts_in_dir(tmp_path / "out")
    assert not errors


def test_tampered_report_extra_field_fails(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "out")
    report = tmp_path / "out" / "enterprise_audit_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["forged_field"] = True
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert validate_report_json(report)


def test_truncated_json_fails(tmp_path: Path) -> None:
    path = tmp_path / "enterprise_audit_outcomes.json"
    path.write_text("[{", encoding="utf-8")
    errors = validate_named_artifact(tmp_path, "enterprise_audit_outcomes.json")
    assert errors


def test_b5_l3_bundle_schemas_validate() -> None:
    root = Path(__file__).resolve().parents[4]
    bundle = (
        root
        / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
        / "qualification_at_b8bb88c9db1060153ec0a6a29bbb79ba16e00078/l3-cli-full"
    )
    if not bundle.is_dir():
        pytest.skip("B5 handoff bundle not present")
    errors = validate_b5_handoff_bundle(bundle)
    assert not errors, errors
