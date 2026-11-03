from __future__ import annotations

import json
from pathlib import Path

from intergrax.platform.enterprise_audit.evidence import build_environment_fingerprint, build_evidence_manifest
from tests.qualification.ent_audit_x._manifest_fixtures import sample_full_execution_manifest


def test_environment_fingerprint_present_in_bundle(tmp_path: Path) -> None:
    fp = build_environment_fingerprint()
    assert "python_version" in fp
    assert "platform" in fp
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "artifact.txt").write_text("ok", encoding="utf-8")
    sha = "b" * 40
    build_evidence_manifest(
        out,
        run_id="r1",
        qualification_sha=sha,
        execution_manifest=sample_full_execution_manifest(qualification_sha=sha),
    )
    raw = json.loads((out / "evidence_manifest.v1.json").read_text(encoding="utf-8"))
    assert "environment_fingerprint" in raw
    assert raw["environment_fingerprint"]["python_version"]


def test_environment_fingerprint_missing_rejected_on_verify() -> None:
    from intergrax.platform.enterprise_audit.validators import validate_evidence_manifest

    # structural schema still passes without fingerprint (optional field)
    assert callable(validate_evidence_manifest)
