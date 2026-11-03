from __future__ import annotations

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditMode, CheckRunOutcome
from intergrax.platform.enterprise_audit.digests import manifest_digest
from intergrax.platform.enterprise_audit.evidence import assert_certifying_publication_eligible
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from tests.qualification.ent_audit_x._manifest_fixtures import sample_full_execution_manifest


def test_publication_rejects_skipped_checks() -> None:
    base = sample_full_execution_manifest()
    manifest = AuditExecutionManifest(
        audit_mode=base.audit_mode,
        qualification_sha=base.qualification_sha,
        audit_execution_sha=base.audit_execution_sha,
        checks_planned=85,
        checks_executed=84,
        checks_skipped=1,
        coverage_label="FULL",
        certifying_run=True,
        run_status="completed",
        check_manifest_digest=manifest_digest(),
        outcomes=base.outcomes,
    )
    with pytest.raises(ValueError, match="checks_skipped"):
        assert_certifying_publication_eligible(manifest)


def test_publication_rejects_failed_outcome_exit_code() -> None:
    checks = load_default_manifest()
    first = checks[0]
    outcomes = tuple(
        CheckRunOutcome(
            check_id=c.check_id,
            executed=True,
            skipped=False,
            exit_code=1 if c.check_id == first.check_id else 0,
        )
        for c in checks
    )
    manifest = AuditExecutionManifest(
        audit_mode=AuditMode.FULL,
        qualification_sha="a" * 40,
        audit_execution_sha="a" * 40,
        checks_planned=85,
        checks_executed=85,
        checks_skipped=0,
        coverage_label="FULL",
        certifying_run=True,
        run_status="completed",
        check_manifest_digest=manifest_digest(),
        outcomes=outcomes,
    )
    with pytest.raises(ValueError, match=first.check_id):
        assert_certifying_publication_eligible(manifest)


def test_verify_evidence_manifest_rejects_traversal_paths(tmp_path) -> None:
    from intergrax.platform.enterprise_audit.evidence import verify_evidence_manifest

    out = tmp_path / "ev"
    out.mkdir()
    (out / "good.txt").write_text("ok", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": "r1",
        "qualification_sha": "a" * 40,
        "certifying_run": True,
        "artifacts": [{"path": "../good.txt", "sha256": "dead"}],
    }
    (out / "evidence_manifest.v1.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    assert verify_evidence_manifest(out) is False
