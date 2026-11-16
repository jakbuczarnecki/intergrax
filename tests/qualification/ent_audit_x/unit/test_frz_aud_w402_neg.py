"""W4 corrective gap tests for FRZ-AUD-01/02/07 (Annex B negatives; no check-manifest edits)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.manifest import load_default_manifest, load_manifest_from_path
from intergrax.platform.enterprise_audit.validators import validate_report_json


def test_ent_audit_runner_missing_neg(tmp_path: Path) -> None:
    """Fail-closed when the check manifest is empty (runner would have zero checks)."""
    manifest_path = tmp_path / "enterprise_audit_check_manifest.v1.json"
    manifest_path.write_text(json.dumps({"checks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="85 checks"):
        load_manifest_from_path(manifest_path)


def test_ent_audit_monolith_script_neg() -> None:
    checks = load_default_manifest()
    entrypoints = {c.entrypoint for c in checks}
    assert len(checks) == 85
    assert len(entrypoints) > 1
    monolith = "scripts/enterprise_audit/run_enterprise_audit.py"
    assert sum(1 for c in checks if c.entrypoint == monolith) < len(checks)


def test_ent_audit_invalid_drift_class_neg(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    sample = (
        root
        / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff/"
        / "qualification_at_b5c2481d79f955fa5f483a9288aee8de4581f953/l3-cli-full/enterprise_audit_report.json"
    )
    if not sample.is_file():
        pytest.skip("historical B5 report fixture unavailable for drift schema neg test")
    payload = json.loads(sample.read_text(encoding="utf-8"))
    findings = list(payload.get("findings") or [])
    if not findings:
        findings = [
            {
                "finding_id": "TEST:1",
                "check_id": "ENT-A-BND-01",
                "severity": "WARNING",
                "category": "architecture",
                "message": "fixture",
                "drift": "NEW",
            }
        ]
    else:
        findings = [dict(findings[0])]
    findings[0]["drift"] = "BOGUS"
    payload = {**payload, "findings": findings}
    bogus_path = tmp_path / "enterprise_audit_report.json"
    bogus_path.write_text(json.dumps(payload), encoding="utf-8")
    errors = validate_report_json(bogus_path)
    assert errors
    joined = "\n".join(errors)
    assert "BOGUS" in joined or "drift" in joined.lower()
