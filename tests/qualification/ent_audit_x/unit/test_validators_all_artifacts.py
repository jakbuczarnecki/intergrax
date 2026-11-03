from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.validators import validate_against_schema, validate_all_artifacts_in_dir


def test_extra_property_fails_closed(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
    }
    errors = validate_against_schema({"a": "ok", "unexpected": 1}, schema)
    assert any("unexpected property" in e for e in errors)


def test_validate_all_artifacts_in_dir_on_run_output(tmp_path: Path) -> None:
    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, _, _ = runner.run(AuditMode.FAST, output_dir=tmp_path / "out")
    errors = validate_all_artifacts_in_dir(tmp_path / "out")
    assert not errors
