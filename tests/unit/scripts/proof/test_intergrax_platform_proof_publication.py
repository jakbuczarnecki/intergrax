# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.proof.intergrax_platform_proof_artifact_verifier import (
    ArtifactVerificationResult,
    ArtifactVerificationStatus,
    verify_platform_proof_artifacts,
)
from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
    ExpectedArtifactKind,
    ExpectedProofArtifact,
)
from scripts.proof.intergrax_platform_proof_discovery import discover_platform_proof_descriptors
from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEvidence,
    ConclusionEvidence,
    EnvironmentEvidence,
    ExecutionMetadataEvidence,
    FailureClassification,
    FailureEvidence,
    ParticipantClass,
    ParticipantEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofIdentityEvidence,
    ProvenanceEvidence,
    ReproductionEvidence,
    sanitized_runtime_report_safe_text,
)
from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    verify_platform_proof_evidence,
)
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    ProofExecutionSpec,
    proof_run_artifact_directory,
    suite_run_artifact_directory,
)
from scripts.proof.intergrax_platform_proof_publication import (
    PublicationStatus,
    apply_canonical_publication,
    canonical_proof_output_directory,
    publish_verified_proof_artifacts,
    should_publish_canonical_output,
)
from scripts.proof.intergrax_proof_contracts import (
    ArtifactVerificationStatus as ProofArtifactVerificationStatus,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
)
from scripts.proof.intergrax_proof_runner import (
    GitMetadata,
    RunnerConfig,
    run_suite,
)

pytestmark = pytest.mark.unit

_FAKE_PROOF_ID = "TEST-PUBLISH-PROOF"
_FAKE_DOMAIN_ROOT = Path("platform_proofs") / "test_domain" / "fake_publish"


def _entry(proof_id: str = _FAKE_PROOF_ID) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        domain="test_domain",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="publish",
        command=ProofArgvCommand(
            executable="python",
            argv=("platform_proofs/test_domain/fake_publish/run_proof.py",),
        ),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def _artifacts() -> tuple[ExpectedProofArtifact, ...]:
    return (
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
            required=True,
        ),
    )


def _spec(
    entry: ProofManifestEntry,
    *,
    package_root: Path | None,
) -> ProofExecutionSpec:
    return ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=True,
        evidence_schema="intergrax.platform_proof_evidence.v1",
        expected_artifacts=_artifacts(),
        package_root=package_root,
    )


def _transport(
    *,
    status: ProofStatus = ProofStatus.PASS,
    exit_code: int | None = 0,
    diagnostic: str = "child_exit_zero",
) -> ProofRunResult:
    return ProofRunResult(
        proof_id=_FAKE_PROOF_ID,
        status=status,
        duration_seconds=1.0,
        exit_code=exit_code,
        diagnostic_summary=diagnostic,
    )


def _write_evidence(
    directory: Path,
    *,
    status: ProofEvidenceExecutionStatus = ProofEvidenceExecutionStatus.PASS,
    execution_id: str = "exec-1",
) -> None:
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
    revision = "unknown"
    failure = None
    if status == ProofEvidenceExecutionStatus.CRASH:
        failure = FailureEvidence(
            classification=FailureClassification.UNKNOWN,
            boundary="provider request",
            message=sanitized_runtime_report_safe_text("simulated crash"),
            completed_milestones=("started",),
            failed_milestone="provider request",
            skipped_not_reached=("evaluator",),
        )
    evidence = PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=_FAKE_PROOF_ID,
            title="title",
            domain="test_domain",
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
            proof_id=_FAKE_PROOF_ID,
            source_revision=revision,
            generated_at=finished,
            execution_id=execution_id,
            artifact_identity=f"artifact-{execution_id}",
        ),
        failure=failure,
        conclusion=ConclusionEvidence(),
    )
    write_evidence_json(evidence, proof_directory=directory)


def _write_candidate_set(
    directory: Path,
    *,
    marker: str = "run-a",
    evidence_status: ProofEvidenceExecutionStatus = ProofEvidenceExecutionStatus.PASS,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_evidence(directory, status=evidence_status, execution_id=marker)
    (directory / "proof-result.json").write_text(
        json.dumps({"marker": marker}),
        encoding="utf-8",
    )
    (directory / "report.html").write_text(
        f"<html><body>{marker}</body></html>",
        encoding="utf-8",
    )


def _verify_and_publish(
    candidate: Path,
    package_root: Path,
    *,
    transport: ProofRunResult | None = None,
) -> ProofRunResult:
    entry = _entry()
    spec = _spec(entry, package_root=package_root)
    transport_result = transport or _transport()
    artifact_summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate,
        spec=spec,
    )
    result = transport_result
    evidence_verification = None
    if artifact_summary.passed:
        evidence_verification = verify_platform_proof_evidence(
            evidence_path=candidate / "evidence.json",
            artifact_root=candidate,
            spec=spec,
            subprocess_result=result,
            expected_source_revision="unknown",
        )
    else:
        result = result.model_copy(update={"status": ProofStatus.FAIL})
    return apply_canonical_publication(
        result,
        transport_result=transport_result,
        execution_spec=spec,
        candidate_directory=candidate,
        artifact_summary=artifact_summary,
        evidence_verification=evidence_verification,
    )


def test_canonical_proof_output_directory_derives_package_root_output(tmp_path: Path) -> None:
    package_root = tmp_path / "platform_proofs" / "tools" / "sample"
    package_root.mkdir(parents=True)
    spec = _spec(_entry(), package_root=package_root)
    assert canonical_proof_output_directory(spec) == (package_root / "output").resolve()


def test_descriptor_without_package_root_fails_closed() -> None:
    spec = _spec(_entry(), package_root=None)
    with pytest.raises(ValueError, match="package_root"):
        canonical_proof_output_directory(spec)


def test_canonical_output_path_cannot_escape_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    spec = _spec(_entry(), package_root=package_root)
    output = canonical_proof_output_directory(spec)
    assert output.resolve().is_relative_to(package_root.resolve())


def test_verified_pass_candidate_publishes_declared_artifacts(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate, marker="run-a")
    _verify_and_publish(candidate, package_root)
    output = package_root / "output"
    assert (output / "evidence.json").is_file()
    assert (output / "proof-result.json").is_file()
    assert (output / "report.html").is_file()


def test_published_bytes_equal_verified_candidate_bytes(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate, marker="byte-check")
    _verify_and_publish(candidate, package_root)
    for name in ("evidence.json", "proof-result.json", "report.html"):
        assert (package_root / "output" / name).read_bytes() == (candidate / name).read_bytes()


def test_second_valid_run_overwrites_canonical_artifacts(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _write_candidate_set(candidate_b, marker="run-b")
    _verify_and_publish(candidate_a, package_root)
    _verify_and_publish(candidate_b, package_root)
    output = package_root / "output"
    assert json.loads((output / "proof-result.json").read_text())["marker"] == "run-b"
    assert "run-b" in (output / "report.html").read_text()
    assert len(list(output.iterdir())) == 3


def test_no_timestamp_directory_under_canonical_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate)
    _verify_and_publish(candidate, package_root)
    output = package_root / "output"
    assert all(not child.name.startswith("202") for child in output.iterdir())
    assert all(child.is_file() for child in output.iterdir())


def test_malformed_evidence_does_not_replace_previous_canonical_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _write_candidate_set(good, marker="run-a")
    _verify_and_publish(good, package_root)
    bad.mkdir()
    bad.joinpath("evidence.json").write_text("{broken", encoding="utf-8")
    bad.joinpath("proof-result.json").write_text('{"marker":"run-b"}', encoding="utf-8")
    bad.joinpath("report.html").write_text("<html></html>", encoding="utf-8")
    _verify_and_publish(bad, package_root)
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-a"


def test_missing_report_does_not_replace_previous_canonical_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    good = tmp_path / "good"
    partial = tmp_path / "partial"
    _write_candidate_set(good, marker="run-a")
    _verify_and_publish(good, package_root)
    partial.mkdir()
    _write_evidence(partial)
    partial.joinpath("proof-result.json").write_text('{"marker":"run-b"}', encoding="utf-8")
    _verify_and_publish(partial, package_root)
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-a"


def test_timeout_does_not_replace_previous_canonical_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    good = tmp_path / "good"
    _write_candidate_set(good, marker="run-a")
    _verify_and_publish(good, package_root)
    timeout = ProofRunResult(
        proof_id=_FAKE_PROOF_ID,
        status=ProofStatus.FAIL,
        duration_seconds=1.0,
        exit_code=None,
        diagnostic_summary="timeout",
    )
    apply_canonical_publication(
        timeout,
        transport_result=timeout,
        execution_spec=_spec(_entry(), package_root=package_root),
        candidate_directory=None,
        artifact_summary=None,
        evidence_verification=None,
    )
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-a"


def test_valid_typed_fail_execution_can_replace_previous_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    pass_candidate = tmp_path / "pass"
    fail_candidate = tmp_path / "fail"
    _write_candidate_set(pass_candidate, marker="run-pass")
    _verify_and_publish(pass_candidate, package_root)
    _write_candidate_set(
        fail_candidate,
        marker="run-fail",
        evidence_status=ProofEvidenceExecutionStatus.FAIL,
    )
    result = _verify_and_publish(
        fail_candidate,
        package_root,
        transport=_transport(
            status=ProofStatus.FAIL,
            exit_code=1,
            diagnostic="child_exit_nonzero",
        ),
    )
    assert result.status == ProofStatus.FAIL
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-fail"


def test_valid_typed_crash_execution_can_replace_previous_output(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    pass_candidate = tmp_path / "pass"
    crash_candidate = tmp_path / "crash"
    _write_candidate_set(pass_candidate, marker="run-pass")
    _verify_and_publish(pass_candidate, package_root)
    _write_candidate_set(
        crash_candidate,
        marker="run-crash",
        evidence_status=ProofEvidenceExecutionStatus.CRASH,
    )
    _verify_and_publish(
        crash_candidate,
        package_root,
        transport=_transport(
            status=ProofStatus.FAIL,
            exit_code=1,
            diagnostic="child_exit_nonzero",
        ),
    )
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-crash"


def test_undeclared_candidate_file_is_not_published(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate)
    candidate.joinpath("extra-secret.txt").write_text("secret", encoding="utf-8")
    _verify_and_publish(candidate, package_root)
    output = package_root / "output"
    assert not (output / "extra-secret.txt").exists()
    assert {path.name for path in output.iterdir()} == {
        "evidence.json",
        "proof-result.json",
        "report.html",
    }


@pytest.mark.skipif(sys.platform == "win32", reason="symlink tests require privileges on Windows")
def test_symlink_candidate_artifact_rejected(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"marker":"outside"}', encoding="utf-8")
    _write_evidence(candidate)
    (candidate / "report.html").write_text("<html></html>", encoding="utf-8")
    (candidate / "proof-result.json").symlink_to(outside)
    _verify_and_publish(candidate, package_root)
    assert not (package_root / "output").exists()


def test_traversal_relative_path_cannot_publish_outside_output_root(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate)
    entry = _entry()
    spec = _spec(entry, package_root=package_root)
    escaped = tmp_path / "escape.json"
    escaped.write_text('{"marker":"escaped"}', encoding="utf-8")
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate,
        spec=spec,
    )
    tampered = tuple(
        ArtifactVerificationResult(
            kind=ExpectedArtifactKind.DOMAIN_RESULT_JSON,
            relative_path="../escape.json",
            required=True,
            status=ArtifactVerificationStatus.PASS,
            resolved_path=escaped,
            diagnostic_code="artifact_verified",
            diagnostic_summary="artifact_verified",
        )
        if result.relative_path == "proof-result.json"
        else result
        for result in summary.results
    )
    from scripts.proof.intergrax_platform_proof_artifact_verifier import (
        ProofArtifactVerificationSummary,
    )

    publication = publish_verified_proof_artifacts(
        candidate_directory=candidate,
        spec=spec,
        artifact_summary=ProofArtifactVerificationSummary(
            passed=True,
            results=tampered,
        ),
    )
    assert publication.status == PublicationStatus.FAILED


def test_staged_publish_failure_does_not_leave_mixed_old_new_artifact_set(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    entry = _entry()
    spec = _spec(entry, package_root=package_root)
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate_b,
        spec=spec,
    )
    copies = 0
    real_copy2 = shutil.copy2

    def _flaky_copy(src, dst, *args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("simulated publish failure")
        return real_copy2(src, dst, *args, **kwargs)

    with patch("scripts.proof.intergrax_platform_proof_publication.shutil.copy2", _flaky_copy):
        publication = publish_verified_proof_artifacts(
            candidate_directory=candidate_b,
            spec=spec,
            artifact_summary=summary,
        )
    assert publication.status == PublicationStatus.FAILED
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-a"


def _publish_with_flaky_rename(
    candidate: Path,
    package_root: Path,
    *,
    fail_on_rename_attempt: int,
    fail_rollback: bool = False,
) -> tuple[object, int]:
    entry = _entry()
    spec = _spec(entry, package_root=package_root)
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate,
        spec=spec,
    )
    rename_attempts = 0
    real_rename = Path.rename

    def _flaky_rename(self: Path, target: Path) -> None:
        nonlocal rename_attempts
        rename_attempts += 1
        if rename_attempts == fail_on_rename_attempt:
            raise OSError("simulated swap failure")
        if fail_rollback and rename_attempts == fail_on_rename_attempt + 1:
            raise OSError("simulated rollback failure")
        return real_rename(self, target)

    with patch.object(Path, "rename", _flaky_rename):
        publication = publish_verified_proof_artifacts(
            candidate_directory=candidate,
            spec=spec,
            artifact_summary=summary,
        )
    return publication, rename_attempts


def test_first_publish_without_previous_output_installs_new(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate, marker="run-first")
    _verify_and_publish(candidate, package_root)
    output = package_root / "output"
    assert output.is_dir()
    assert json.loads((output / "proof-result.json").read_text())["marker"] == "run-first"


def test_swap_failure_restores_old_byte_for_byte(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    old_bytes = {
        name: (package_root / "output" / name).read_bytes()
        for name in ("evidence.json", "proof-result.json", "report.html")
    }
    _write_candidate_set(candidate_b, marker="run-b")
    publication, _ = _publish_with_flaky_rename(
        candidate_b,
        package_root,
        fail_on_rename_attempt=2,
    )
    assert publication.status == PublicationStatus.FAILED
    output = package_root / "output"
    for name, content in old_bytes.items():
        assert (output / name).read_bytes() == content


def test_swap_failure_preserves_all_canonical_files_from_old_set(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    publication, _ = _publish_with_flaky_rename(
        candidate_b,
        package_root,
        fail_on_rename_attempt=2,
    )
    assert publication.status == PublicationStatus.FAILED
    assert json.loads((package_root / "output" / "proof-result.json").read_text())["marker"] == "run-a"
    assert "run-a" in (package_root / "output" / "report.html").read_text()


def test_swap_failure_leaves_no_mixed_artifact_set(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    publication, _ = _publish_with_flaky_rename(
        candidate_b,
        package_root,
        fail_on_rename_attempt=2,
    )
    assert publication.status == PublicationStatus.FAILED
    output = package_root / "output"
    assert {path.name for path in output.iterdir()} == {
        "evidence.json",
        "proof-result.json",
        "report.html",
    }
    for name in ("evidence.json", "proof-result.json", "report.html"):
        assert "run-b" not in (output / name).read_bytes().decode("utf-8", errors="replace")


def test_successful_replacement_removes_backup_directory(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    _verify_and_publish(candidate_b, package_root)
    backups = list(package_root.glob(".proof-publish-backup-*"))
    assert backups == []


def test_failed_replacement_cleans_staging_directories(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    publication, _ = _publish_with_flaky_rename(
        candidate_b,
        package_root,
        fail_on_rename_attempt=2,
    )
    assert publication.status == PublicationStatus.FAILED
    staging_dirs = [
        path
        for path in package_root.iterdir()
        if path.name.startswith(".proof-publish-")
        and not path.name.startswith(".proof-publish-backup-")
    ]
    assert staging_dirs == []


def test_rollback_failure_reports_explicit_diagnostic(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_a, marker="run-a")
    _verify_and_publish(candidate_a, package_root)
    _write_candidate_set(candidate_b, marker="run-b")
    publication, _ = _publish_with_flaky_rename(
        candidate_b,
        package_root,
        fail_on_rename_attempt=2,
        fail_rollback=True,
    )
    assert publication.status == PublicationStatus.FAILED
    assert publication.diagnostic_code == "canonical_output_publish_rollback_failed"


def test_old_removed_artifact_does_not_remain_stale_after_replacement(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate)
    candidate.joinpath("legacy.txt").write_text("legacy", encoding="utf-8")
    entry = _entry()
    spec_with_legacy = ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=False,
        expected_artifacts=_artifacts()
        + (
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.OTHER,
                relative_path="legacy.txt",
                required=True,
            ),
        ),
        package_root=package_root,
    )
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate,
        spec=spec_with_legacy,
    )
    publish_verified_proof_artifacts(
        candidate_directory=candidate,
        spec=spec_with_legacy,
        artifact_summary=summary,
    )
    assert (package_root / "output" / "legacy.txt").exists()

    candidate_b = tmp_path / "candidate-b"
    _write_candidate_set(candidate_b, marker="no-legacy")
    spec_current = _spec(entry, package_root=package_root)
    summary_b = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate_b,
        spec=spec_current,
    )
    publish_verified_proof_artifacts(
        candidate_directory=candidate_b,
        spec=spec_current,
        artifact_summary=summary_b,
    )
    output = package_root / "output"
    assert not (output / "legacy.txt").exists()
    assert {path.name for path in output.iterdir()} == {
        "evidence.json",
        "proof-result.json",
        "report.html",
    }


def test_publication_failure_prevents_pass(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    candidate = tmp_path / "candidate"
    _write_candidate_set(candidate)
    entry = _entry()
    spec = _spec(entry, package_root=package_root)
    summary = verify_platform_proof_artifacts(
        proof_artifact_directory=candidate,
        spec=spec,
    )
    evidence_verification = verify_platform_proof_evidence(
        evidence_path=candidate / "evidence.json",
        artifact_root=candidate,
        spec=spec,
        subprocess_result=_transport(),
        expected_source_revision="unknown",
    )
    with patch(
        "scripts.proof.intergrax_platform_proof_publication.publish_verified_proof_artifacts",
        return_value=__import__(
            "scripts.proof.intergrax_platform_proof_publication",
            fromlist=["PublicationResult"],
        ).PublicationResult(
            status=PublicationStatus.FAILED,
            published_output_path=None,
            diagnostic_code="canonical_output_publish_failed",
            diagnostic_summary="canonical_output_publish_failed",
        ),
    ):
        result = apply_canonical_publication(
            _transport(),
            transport_result=_transport(),
            execution_spec=spec,
            candidate_directory=candidate,
            artifact_summary=summary,
            evidence_verification=evidence_verification,
        )
    assert result.status == ProofStatus.FAIL
    assert result.diagnostic_summary == "canonical_output_publish_failed"


def _descriptor_payload(*, proof_id: str, entrypoint: str, run_mode: str) -> dict[str, object]:
    return {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": "CONFORMANCE",
        "proof_id": proof_id,
        "title": proof_id,
        "domain": "test_domain",
        "proof_kind": "publish",
        "mechanisms_exercised": ["tools.sample_mechanism"],
        "package_version": "1.0.0",
        "profiles": ["quick"],
        "command": {"executable": "uv", "argv": ["run", "python", entrypoint]},
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
        "evidence_required": True,
        "evidence_schema": "intergrax.platform_proof_evidence.v1",
        "report_required": True,
        "expected_artifacts": [
            {"kind": "EVIDENCE_JSON", "relative_path": "evidence.json", "required": True},
            {"kind": "DOMAIN_RESULT_JSON", "relative_path": "proof-result.json", "required": True},
            {"kind": "REPORT_HTML", "relative_path": "report.html", "required": True},
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

def _build_evidence(status: ProofEvidenceExecutionStatus, execution_id: str) -> PlatformProofEvidence:
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
            domain="test_domain",
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
            execution_id=execution_id,
            artifact_identity=f"artifact-{{execution_id}}",
        ),
    )

def main() -> int:
    artifact_dir = os.environ.get("{INTERGRAX_PROOF_ARTIFACT_DIR_ENV}", "").strip()
    if not artifact_dir:
        print("missing artifact dir", file=sys.stderr)
        return 2
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    execution_id = os.environ.get("INTERGRAX_TEST_EXECUTION_ID", "exec-1")
    if RUN_MODE == "pass":
        status = ProofEvidenceExecutionStatus.PASS
        write_evidence_json(_build_evidence(status, execution_id), proof_directory=directory)
        directory.joinpath("proof-result.json").write_text(
            json.dumps({{"marker": execution_id}}),
            encoding="utf-8",
        )
        directory.joinpath("report.html").write_text(
            f"<html><body>{{execution_id}}</body></html>",
            encoding="utf-8",
        )
        return 0
    if RUN_MODE == "missing_report":
        write_evidence_json(
            _build_evidence(ProofEvidenceExecutionStatus.PASS, execution_id),
            proof_directory=directory,
        )
        directory.joinpath("proof-result.json").write_text(
            json.dumps({{"marker": execution_id}}),
            encoding="utf-8",
        )
        return 0
    if RUN_MODE == "typed_fail":
        write_evidence_json(
            _build_evidence(ProofEvidenceExecutionStatus.FAIL, execution_id),
            proof_directory=directory,
        )
        directory.joinpath("proof-result.json").write_text(
            json.dumps({{"marker": execution_id}}),
            encoding="utf-8",
        )
        directory.joinpath("report.html").write_text(
            f"<html><body>{{execution_id}}</body></html>",
            encoding="utf-8",
        )
        return 1
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_fake_proof_package(repo_root: Path, *, run_mode: str = "pass") -> Path:
    package = repo_root / _FAKE_DOMAIN_ROOT
    package.mkdir(parents=True, exist_ok=True)
    entrypoint = "platform_proofs/test_domain/fake_publish/run_proof.py"
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
    return package


def _isolate_fake_proof_discovery(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_mode: str = "pass",
) -> Path:
    monkeypatch.setattr(
        "scripts.proof.intergrax_platform_proof_execution.build_manifest_entries",
        lambda: (),
    )
    package = _write_fake_proof_package(repo_root, run_mode=run_mode)
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    fake_only = tuple(
        item for item in discovered if item.manifest_entry.proof_id == _FAKE_PROOF_ID
    )
    monkeypatch.setattr(
        "scripts.proof.intergrax_platform_proof_execution.discover_platform_proof_descriptors",
        lambda repo_root=None, **kwargs: fake_only,
    )
    return package


def _cleanup_fake_packages(repo_root: Path) -> None:
    shutil.rmtree(repo_root / _FAKE_DOMAIN_ROOT, ignore_errors=True)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_suite_publishes_verified_tools_fake_candidate_to_package_output(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    monkeypatch.setattr(
        "scripts.proof.intergrax_proof_runner.read_git_metadata",
        lambda repo_root: GitMetadata(commit_sha="unknown", dirty=False),
    )
    monkeypatch.setenv("INTERGRAX_TEST_SOURCE_REVISION", "unknown")
    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "exec-first")

    receipt, receipt_path = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    result = receipt.results[0]
    assert result.status == ProofStatus.PASS
    assert result.artifact_verification_status == ProofArtifactVerificationStatus.PASS
    assert receipt_path is not None
    assert receipt_path.parent == suite_run_artifact_directory(repo_root, receipt.suite_run_id)
    assert receipt_path.name == "suite-receipt.json"

    canonical = package / "output"
    assert (canonical / "evidence.json").is_file()
    assert (canonical / "proof-result.json").is_file()
    assert (canonical / "report.html").is_file()
    assert json.loads((canonical / "proof-result.json").read_text())["marker"] == "exec-first"

    candidate = proof_run_artifact_directory(
        repo_root,
        receipt.suite_run_id,
        _FAKE_PROOF_ID,
    )
    assert candidate.is_dir()
    assert candidate != canonical

    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "exec-second")
    receipt_b, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    assert receipt_b.results[0].status == ProofStatus.PASS
    assert json.loads((canonical / "proof-result.json").read_text())["marker"] == "exec-second"
    assert len(list(canonical.iterdir())) == 3
    _cleanup_fake_packages(repo_root)


def test_suite_negative_integration_missing_report_preserves_previous_output(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    monkeypatch.setattr(
        "scripts.proof.intergrax_proof_runner.read_git_metadata",
        lambda repo_root: GitMetadata(commit_sha="unknown", dirty=False),
    )
    monkeypatch.setenv("INTERGRAX_TEST_SOURCE_REVISION", "unknown")
    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "run-a")
    run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    run_a_bytes = {
        name: (package / "output" / name).read_bytes()
        for name in ("evidence.json", "proof-result.json", "report.html")
    }

    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="missing_report")
    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "run-b")
    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    assert receipt.results[0].status == ProofStatus.FAIL
    for name, content in run_a_bytes.items():
        assert (package / "output" / name).read_bytes() == content
    _cleanup_fake_packages(repo_root)


def test_suite_fail_publication_replaces_pass_with_truthful_fail(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    monkeypatch.setattr(
        "scripts.proof.intergrax_proof_runner.read_git_metadata",
        lambda repo_root: GitMetadata(commit_sha="unknown", dirty=False),
    )
    monkeypatch.setenv("INTERGRAX_TEST_SOURCE_REVISION", "unknown")
    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "run-pass")
    run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    assert json.loads((package / "output" / "proof-result.json").read_text())["marker"] == "run-pass"

    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="typed_fail")
    monkeypatch.setenv("INTERGRAX_TEST_EXECUTION_ID", "run-fail")
    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    assert receipt.results[0].status == ProofStatus.FAIL
    assert json.loads((package / "output" / "proof-result.json").read_text())["marker"] == "run-fail"
    _cleanup_fake_packages(repo_root)


def test_canonical_output_is_not_gitignored(repo_root: Path) -> None:
    target = repo_root / _FAKE_DOMAIN_ROOT / "output" / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("<html></html>", encoding="utf-8")
    completed = subprocess.run(
        ["git", "check-ignore", "-v", str(target)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert completed.returncode == 1


def test_should_publish_requires_verified_artifacts_and_evidence(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    spec = _spec(_entry(), package_root=package_root)
    transport = _transport()
    assert should_publish_canonical_output(
        transport_result=transport,
        execution_spec=spec,
        artifact_summary=None,
        evidence_verification=None,
    ) is False
