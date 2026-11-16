from __future__ import annotations

import json
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.baseline import (
    EnterpriseAuditBaseline,
    assert_baseline_manifest_digest_current,
    baseline_path,
    candidate_baseline_path,
    load_candidate_baseline,
    promote_accepted_baseline,
    save_candidate_baseline,
)
from intergrax.platform.enterprise_audit.digests import manifest_digest

_DIGEST = "0fe966e43a10a561c38222b6cff30269e906633e61862d76aa29c1775e691b8b"
_SHA = "f" * 40


def test_load_candidate_baseline_preserves_check_manifest_digest(tmp_path: Path) -> None:
    path = candidate_baseline_path(tmp_path)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_baseline_sha": _SHA,
                "check_manifest_digest": _DIGEST,
                "findings": {},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_candidate_baseline(path)
    assert loaded is not None
    assert loaded.check_manifest_digest == _DIGEST


def test_promote_copies_check_manifest_digest_to_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_candidate_baseline(
        candidate_baseline_path(tmp_path),
        EnterpriseAuditBaseline(
            accepted_baseline_sha=_SHA,
            findings={},
            check_manifest_digest=manifest_digest(),
        ),
    )
    od = tmp_path / "od.json"

    def _stub_verify(*_args, **_kwargs):
        return {"accepted_baseline_sha": _SHA, "test_execution_sha": "c" * 40}

    monkeypatch.setattr(
        "intergrax.platform.enterprise_audit.baseline.verify_operator_decision_record",
        _stub_verify,
    )
    monkeypatch.setattr(
        "intergrax.platform.enterprise_audit.validators.verify_closed_evidence_bundle",
        lambda _d: [],
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evidence_manifest.v1.json").write_text(json.dumps({"qualification_sha": "c" * 40}), encoding="utf-8")
    out = promote_accepted_baseline(
        tmp_path,
        od_record_path=od,
        verify_closed_bundle_dir=bundle,
        repo_root=tmp_path,
    )
    accepted = json.loads(out.read_text(encoding="utf-8"))
    assert accepted["check_manifest_digest"] == manifest_digest()
    assert accepted["check_manifest_digest"] == _DIGEST


def test_promote_rejects_candidate_without_check_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_candidate_baseline(
        candidate_baseline_path(tmp_path),
        EnterpriseAuditBaseline(accepted_baseline_sha=_SHA, findings={}),
    )
    od = tmp_path / "od.json"

    def _stub_verify(*_args, **_kwargs):
        return {"accepted_baseline_sha": _SHA, "test_execution_sha": "c" * 40}

    monkeypatch.setattr(
        "intergrax.platform.enterprise_audit.baseline.verify_operator_decision_record",
        _stub_verify,
    )
    monkeypatch.setattr(
        "intergrax.platform.enterprise_audit.validators.verify_closed_evidence_bundle",
        lambda _d: [],
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evidence_manifest.v1.json").write_text(json.dumps({"qualification_sha": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="check_manifest_digest"):
        promote_accepted_baseline(
            tmp_path,
            od_record_path=od,
            verify_closed_bundle_dir=bundle,
            repo_root=tmp_path,
        )
    assert not baseline_path(tmp_path).is_file()


def test_assert_baseline_manifest_digest_current_rejects_empty() -> None:
    baseline = EnterpriseAuditBaseline(accepted_baseline_sha=_SHA, findings={})
    with pytest.raises(ValueError, match="missing check_manifest_digest"):
        assert_baseline_manifest_digest_current(baseline, manifest_digest())
