# Â© Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import shutil
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_artifact_verifier import (
    ArtifactVerificationStatus,
    ArtifactVerificationResult,
    apply_artifact_verification,
    verify_platform_proof_artifacts,
)
from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
    ExpectedArtifactKind,
    ExpectedProofArtifact,
    PlatformProofDescriptor,
)
from scripts.proof.intergrax_platform_proof_discovery import discover_platform_proof_descriptors
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    ProofExecutionSpec,
)
from scripts.proof.intergrax_proof_contracts import (
    ArtifactVerificationStatus as ProofArtifactVerificationStatus,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
    SuiteOverallStatus,
    SuiteReceipt,
)
from scripts.proof.intergrax_proof_runner import (
    GitMetadata,
    RunnerConfig,
    execute_proof,
    run_suite,
)

pytestmark = pytest.mark.unit

_FAKE_PROOF_ID = "TEST-ARTIFACT-VERIFY-PROOF"
_FAKE_DOMAIN_ROOT = Path("platform_proofs") / "test_artifact_verify"


def _entry(proof_id: str = _FAKE_PROOF_ID) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="artifact_verify",
        command=ProofArgvCommand(
            executable="python",
            argv=(
                "platform_proofs/test_artifact_verify/fake_proof/run_proof.py",
            ),
        ),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def _artifacts_spec(
    entry: ProofManifestEntry,
    *,
    artifacts: tuple[ExpectedProofArtifact, ...] | None = None,
    evidence_required: bool = True,
) -> ProofExecutionSpec:
    if artifacts is None:
        artifacts = (
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.EVIDENCE_JSON,
                relative_path="evidence.json",
                required=True,
            ),
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.REPORT_HTML,
                relative_path="report.html",
                required=False,
            ),
        )
    return ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=evidence_required,
        evidence_schema="intergrax.platform_proof_evidence.v2",
        expected_artifacts=artifacts,
    )


def _transport(
    *,
    status: ProofStatus = ProofStatus.PASS,
    exit_code: int | None = 0,
) -> ProofRunResult:
    return ProofRunResult(
        proof_id=_FAKE_PROOF_ID,
        status=status,
        duration_seconds=1.0,
        exit_code=exit_code,
        diagnostic_summary="child_exit_zero" if status == ProofStatus.PASS else "child_exit_nonzero",
    )


def _verify(
    artifact_dir: Path,
    spec: ProofExecutionSpec,
) -> tuple[object, ProofRunResult]:
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=artifact_dir,
        spec=spec,
    )
    final = apply_artifact_verification(_transport(), summary)
    return summary, final


def _minimal_descriptor_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": "CONFORMANCE",
        "proof_id": "TOOLS-SAMPLE",
        "title": "sample",
        "domains_exercised": ["tools"],
        "proof_kind": "sample",
        "mechanisms_exercised": ["tools.sample_mechanism"],
        "package_version": "1.0.0",
        "profiles": ["full"],
        "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
        "evidence_required": False,
        "report_required": False,
    }
    base.update(overrides)
    return base


def test_required_artifacts_exist_pass(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"ok": true}', encoding="utf-8")
    (artifact_dir / "proof-result.json").write_text('{"result": "pass"}', encoding="utf-8")

    summary, final = _verify(artifact_dir, _artifacts_spec(_entry()))
    assert summary.passed is True
    assert final.status == ProofStatus.PASS
    assert final.artifact_verification_status == ProofArtifactVerificationStatus.PASS


def test_missing_required_artifact_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"ok": true}', encoding="utf-8")

    summary, final = _verify(artifact_dir, _artifacts_spec(_entry()))
    assert summary.passed is False
    assert final.status == ProofStatus.FAIL
    assert "proof-result.json" in final.artifact_diagnostic


def test_optional_artifact_missing_allowed(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"ok": true}', encoding="utf-8")
    (artifact_dir / "proof-result.json").write_text('{"result": "pass"}', encoding="utf-8")

    summary, _ = _verify(artifact_dir, _artifacts_spec(_entry()))
    report = next(r for r in summary.results if r.relative_path == "report.html")
    assert report.status.value == "OPTIONAL_MISSING"


def test_optional_artifact_exists_empty_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"ok": true}', encoding="utf-8")
    (artifact_dir / "proof-result.json").write_text('{"result": "pass"}', encoding="utf-8")
    (artifact_dir / "report.html").write_text("", encoding="utf-8")

    summary, final = _verify(artifact_dir, _artifacts_spec(_entry()))
    assert summary.passed is False
    assert final.status == ProofStatus.FAIL


def test_artifact_path_outside_root_fails(tmp_path: Path) -> None:
    from scripts.proof.intergrax_platform_proof_artifact_verifier import _resolve_artifact_path

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    resolved, error = _resolve_artifact_path(artifact_dir, "../outside.txt")
    assert resolved is None
    assert error == "artifact_path_outside_root"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink tests require elevated privileges on Windows")
def test_symlink_escape_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = artifact_dir / "linked.txt"
    link.symlink_to(outside)

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.OTHER,
                relative_path="linked.txt",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is False
    assert final.status == ProofStatus.FAIL


def test_directory_instead_of_file_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").mkdir()

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is False
    assert "directory" in final.artifact_diagnostic


def test_empty_required_file_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_text("", encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is False
    assert "empty" in final.artifact_diagnostic


@pytest.mark.skipif(sys.platform == "win32", reason="chmod unreadable unreliable on Windows")
def test_unreadable_file_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    path = artifact_dir / "proof-result.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    path.chmod(0o000)

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    try:
        summary, final = _verify(artifact_dir, spec)
        assert summary.passed is False
        assert final.status == ProofStatus.FAIL
    finally:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_valid_domain_result_json_passes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_text('{"status": "ok"}', encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is True
    assert final.status == ProofStatus.PASS


def test_malformed_domain_result_json_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_text("{broken", encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is False
    assert "valid JSON" in final.artifact_diagnostic


def test_non_utf8_domain_result_json_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_bytes(b"\xff\xfe")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is False
    assert "UTF-8" in final.artifact_diagnostic


def test_report_html_optional_missing_allowed(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_text('{"ok": true}', encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.REPORT_HTML,
                relative_path="report.html",
                required=False,
            ),
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                relative_path="proof-result.json",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is True
    assert final.status == ProofStatus.PASS


def test_report_html_existing_passes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "report.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.REPORT_HTML,
                relative_path="report.html",
                required=False,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is True
    assert final.status == ProofStatus.PASS


def test_other_non_empty_passes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "notes.txt").write_text("artifact", encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.OTHER,
                relative_path="notes.txt",
                required=True,
            ),
        ),
    )
    summary, final = _verify(artifact_dir, spec)
    assert summary.passed is True
    assert final.status == ProofStatus.PASS


def test_duplicate_relative_path_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        PlatformProofDescriptor.model_validate(
            _minimal_descriptor_payload(
                expected_artifacts=[
                    {
                        "kind": "OTHER",
                        "relative_path": "dup.txt",
                        "required": True,
                    },
                    {
                        "kind": "OTHER",
                        "relative_path": "dup.txt",
                        "required": False,
                    },
                ]
            )
        )


def test_evidence_required_without_required_evidence_json_rejected() -> None:
    with pytest.raises(ValidationError, match="EVIDENCE_JSON"):
        PlatformProofDescriptor.model_validate(
            _minimal_descriptor_payload(
                evidence_required=True,
                expected_artifacts=[],
            )
        )


def test_report_required_without_required_report_html_rejected() -> None:
    with pytest.raises(ValidationError, match="REPORT_HTML"):
        PlatformProofDescriptor.model_validate(
            _minimal_descriptor_payload(
                report_required=True,
                expected_artifacts=[],
            )
        )


def test_contradictory_required_evidence_vs_flag_rejected() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        PlatformProofDescriptor.model_validate(
            _minimal_descriptor_payload(
                evidence_required=False,
                expected_artifacts=[
                    {
                        "kind": "EVIDENCE_JSON",
                        "relative_path": "evidence.json",
                        "required": True,
                    }
                ],
            )
        )


def test_contradictory_required_report_vs_flag_rejected() -> None:
    with pytest.raises(ValidationError, match="contradicts"):
        PlatformProofDescriptor.model_validate(
            _minimal_descriptor_payload(
                report_required=False,
                expected_artifacts=[
                    {
                        "kind": "REPORT_HTML",
                        "relative_path": "report.html",
                        "required": True,
                    }
                ],
            )
        )


def test_child_exit_nonzero_remains_fail_with_valid_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"ok": true}', encoding="utf-8")
    (artifact_dir / "proof-result.json").write_text('{"ok": true}', encoding="utf-8")

    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=artifact_dir,
        spec=_artifacts_spec(_entry()),
    )
    final = apply_artifact_verification(_transport(status=ProofStatus.FAIL, exit_code=1), summary)
    assert summary.passed is True
    assert final.status == ProofStatus.FAIL
    assert final.exit_code == 1


def test_legacy_spec_without_artifacts_skips_verification(tmp_path: Path) -> None:
    entry = _entry()
    spec = ProofExecutionSpec(manifest_entry=entry)
    result = execute_proof(entry, repo_root=tmp_path, execution_spec=spec)
    assert result.artifact_verification_status is None


def test_suite_receipt_reflects_artifact_fail_without_contents(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "evidence.json").write_text('{"secret": "value"}', encoding="utf-8")

    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=artifact_dir,
        spec=_artifacts_spec(_entry()),
    )
    final = apply_artifact_verification(_transport(), summary)
    receipt = SuiteReceipt(
        suite_run_id="suite-1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        git_commit_sha="abc123",
        git_dirty=False,
        profile=ProofProfile.QUICK,
        platform="linux",
        python_version="3.12",
        overall_status=SuiteOverallStatus.FAIL,
        results=(final,),
        passed_count=0,
        failed_count=1,
        blocked_count=0,
        skipped_count=0,
    )
    serialized = json.dumps(receipt.model_dump(mode="json"))
    assert final.status == ProofStatus.FAIL
    assert '"secret"' not in serialized
    assert "artifact_verification_status" in serialized


def _descriptor_payload(
    *,
    proof_id: str,
    entrypoint: str,
    run_mode: str = "pass",
) -> dict[str, object]:
    return {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": "CONFORMANCE",
        "proof_id": proof_id,
        "title": proof_id,
        "domains_exercised": ["test_artifact_verify"],
        "proof_kind": "artifact_verify",
        "mechanisms_exercised": ["tools.sample_mechanism"],
        "package_version": "1.0.0",
        "profiles": ["quick"],
        "command": {
            "executable": "uv",
            "argv": ["run", "python", entrypoint],
        },
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
        "evidence_required": True,
        "evidence_schema": "intergrax.platform_proof_evidence.v2",
        "expected_artifacts": [
            {
                "kind": "EVIDENCE_JSON",
                "relative_path": "evidence.json",
                "required": True,
            },
            {
                "kind": "DOMAIN_RESULT_JSON",
                "relative_path": "proof-result.json",
                "required": True,
            },
            {
                "kind": "REPORT_HTML",
                "relative_path": "report.html",
                "required": False,
            },
        ],
        "tags": ["run_mode", run_mode],
    }


def _fake_run_proof_source(run_mode: str) -> str:
    return f'''
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEvidence,
    EnvironmentEvidence,
    ExecutionMetadataEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofIdentityEvidence,
    ProvenanceEvidence,
    ReproductionEvidence,
    ParticipantClass,
    ParticipantEvidence,
)
from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_proof_contracts import ProofProfile

PROOF_ID = "{_FAKE_PROOF_ID}"
RUN_MODE = "{run_mode}"

def _build_evidence(status: ProofEvidenceExecutionStatus) -> PlatformProofEvidence:
    started = datetime(2026, 8, 22, 8, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 8, 22, 8, 1, 0, tzinfo=UTC)
    participant = ParticipantEvidence(
        participant_id="p1",
        name="n",
        implementation="i",
        version_or_model="v",
        role="r",
        participant_class=ParticipantClass.PLATFORM,
    )
    architecture = ArchitectureEvidence(participants=(participant,))
    revision = os.environ.get("INTERGRAX_TEST_SOURCE_REVISION", "unknown")
    return PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=PROOF_ID,
            title="title",
            domains_exercised=("test_artifact_verify",),
            proof_version="1.0.0",
            source_revision=revision,
            execution_profile=ProofProfile.QUICK,
        ),
        execution=ExecutionMetadataEvidence(
            status=status,
            started_at=started,
            finished_at=finished,
            platform="linux",
        ),
        claim=ProofClaimEvidence(
            claim="claim",
            user_relevance="relevance",
            success_criteria=("ok",),
            falsification_criteria=("fail",),
            excluded_claims=("none",),
        ),
        architecture=architecture,
        participants=architecture.participants,
        environment=EnvironmentEvidence(),
        reproduction=ReproductionEvidence(
            source_revision=revision,
            command="python run_proof.py",
        ),
        provenance=ProvenanceEvidence(
            proof_id=PROOF_ID,
            source_revision=revision,
            generated_at=finished,
            execution_id="exec-1",
            artifact_identity="artifact-1",
        ),
    )

def main() -> int:
    artifact_dir = os.environ.get("{INTERGRAX_PROOF_ARTIFACT_DIR_ENV}", "").strip()
    if not artifact_dir:
        print("missing artifact dir", file=sys.stderr)
        return 2
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    status = ProofEvidenceExecutionStatus.PASS
    write_evidence_json(_build_evidence(status), proof_directory=directory)
    if RUN_MODE == "pass":
        directory.joinpath("proof-result.json").write_text(
            json.dumps({{"status": "pass"}}),
            encoding="utf-8",
        )
        return 0
    if RUN_MODE == "missing_result":
        return 0
    if RUN_MODE == "broken_result":
        directory.joinpath("proof-result.json").write_text("{{broken", encoding="utf-8")
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_fake_proof_package(
    repo_root: Path,
    *,
    run_mode: str = "pass",
) -> None:
    package = repo_root / _FAKE_DOMAIN_ROOT / "fake_proof"
    package.mkdir(parents=True, exist_ok=True)
    entrypoint = "platform_proofs/test_artifact_verify/fake_proof/run_proof.py"
    (package / "run_proof.py").write_text(
        _fake_run_proof_source(run_mode=run_mode),
        encoding="utf-8",
    )
    (package / PROOF_DESCRIPTOR_FILENAME).write_text(
        json.dumps(
            _descriptor_payload(
                proof_id=_FAKE_PROOF_ID,
                entrypoint=entrypoint,
                run_mode=run_mode,
            )
        ),
        encoding="utf-8",
    )


def _empty_static_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.proof.intergrax_platform_proof_execution.build_manifest_entries",
        lambda: (),
    )


def _isolate_fake_proof_discovery(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_mode: str = "pass",
) -> None:
    _empty_static_manifest(monkeypatch)
    _write_fake_proof_package(repo_root, run_mode=run_mode)
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    fake_only = tuple(
        item for item in discovered if item.manifest_entry.proof_id == _FAKE_PROOF_ID
    )
    monkeypatch.setattr(
        "scripts.proof.intergrax_platform_proof_execution.discover_platform_proof_descriptors",
        lambda repo_root=None, **kwargs: fake_only,
    )


def _cleanup_fake_packages(repo_root: Path) -> None:
    shutil.rmtree(repo_root / _FAKE_DOMAIN_ROOT, ignore_errors=True)


def _use_unknown_git_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.proof.intergrax_proof_runner.read_git_metadata",
        lambda repo_root: GitMetadata(commit_sha="unknown", dirty=False),
    )
    monkeypatch.setenv("INTERGRAX_TEST_SOURCE_REVISION", "unknown")


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_runner_integration_all_required_artifacts_pass(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    _use_unknown_git_revision(monkeypatch)

    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    result = receipt.results[0]
    assert result.status == ProofStatus.PASS
    assert result.artifact_verification_status == ProofArtifactVerificationStatus.PASS
    assert receipt.overall_status == SuiteOverallStatus.PASS
    _cleanup_fake_packages(repo_root)


def test_runner_integration_missing_required_artifact_fails(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="missing_result")
    _use_unknown_git_revision(monkeypatch)

    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    result = receipt.results[0]
    assert result.status == ProofStatus.FAIL
    assert result.artifact_verification_status == ProofArtifactVerificationStatus.FAIL
    assert "proof-result.json" in result.artifact_diagnostic
    assert receipt.overall_status == SuiteOverallStatus.FAIL
    _cleanup_fake_packages(repo_root)


def test_runner_integration_broken_json_fails_despite_valid_evidence(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="broken_result")
    _use_unknown_git_revision(monkeypatch)

    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    result = receipt.results[0]
    assert result.status == ProofStatus.FAIL
    assert "valid JSON" in result.artifact_diagnostic
    _cleanup_fake_packages(repo_root)


def test_dry_run_does_not_verify_artifacts(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch)
    receipt, _ = run_suite(
        RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root, dry_run=True)
    )
    result = receipt.results[0]
    assert result.status == ProofStatus.SKIPPED_PROFILE
    assert result.artifact_verification_status is None
    _cleanup_fake_packages(repo_root)


def test_blocked_before_execution_does_not_verify_artifacts(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch)
    monkeypatch.delenv("INTERGRAX_SLACK_APP_TOKEN", raising=False)
    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.LIVE, repo_root=repo_root))
    slack = next(
        (
            item
            for item in receipt.results
            if item.proof_id == "SLACK-CONVERSATION-LIVE"
        ),
        None,
    )
    if slack is not None:
        assert slack.status == ProofStatus.BLOCKED_ENVIRONMENT
        assert slack.artifact_verification_status is None
    _cleanup_fake_packages(repo_root)


def test_verification_order_is_descriptor_declaration(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "a.txt").write_text("a", encoding="utf-8")
    (artifact_dir / "b.txt").write_text("b", encoding="utf-8")

    spec = _artifacts_spec(
        _entry(),
        artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.OTHER,
                relative_path="a.txt",
                required=True,
            ),
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.OTHER,
                relative_path="b.txt",
                required=True,
            ),
        ),
    )
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=artifact_dir,
        spec=spec,
    )
    assert [result.relative_path for result in summary.results] == ["a.txt", "b.txt"]


def test_artifact_results_are_typed_not_dicts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "proof-result.json").write_text('{"ok": true}', encoding="utf-8")

    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=artifact_dir,
        spec=_artifacts_spec(
            _entry(),
            artifacts=(
                ExpectedProofArtifact(
                    kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
                    relative_path="proof-result.json",
                    required=True,
                ),
            ),
        ),
    )
    assert isinstance(summary.results[0], ArtifactVerificationResult)
