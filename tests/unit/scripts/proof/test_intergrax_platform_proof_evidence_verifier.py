# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
)
from scripts.proof.intergrax_platform_proof_evidence import (
    PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
    ArchitectureEvidence,
    EnvironmentEvidence,
    EvaluatorCheckEvidence,
    EvaluatorSummaryEvidence,
    ExecutionMetadataEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofIdentityEvidence,
    ProvenanceEvidence,
    ReproductionEvidence,
    ScenarioEvidence,
)
from scripts.proof.intergrax_platform_proof_evidence_io import (
    EVIDENCE_FILENAME,
    write_evidence_json,
)
from scripts.proof.intergrax_platform_proof_discovery import discover_platform_proof_descriptors
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    EvidenceVerificationResult,
    EvidenceVerificationStatus,
    apply_evidence_verification,
    resolve_expected_evidence_path,
    verify_platform_proof_evidence,
)
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    ProofExecutionSpec,
    proof_run_artifact_directory,
)
from scripts.proof.intergrax_proof_contracts import (
    EvidenceVerificationStatus as ContractEvidenceStatus,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
    SuiteReceipt,
)
from scripts.proof.intergrax_proof_runner import (
    GitMetadata,
    RunnerConfig,
    execute_proof,
    run_suite,
)

pytestmark = pytest.mark.unit

_FAKE_PROOF_ID = "TEST-EVIDENCE-VERIFY-PROOF"
_FAKE_DOMAIN_ROOT = Path("platform_proofs") / "test_evidence_verify"


def _entry(proof_id: str = _FAKE_PROOF_ID) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        domain="test_evidence_verify",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="evidence_verify",
        command=ProofArgvCommand(
            executable="python",
            argv=(
                "platform_proofs/test_evidence_verify/fake_proof/run_proof.py",
            ),
        ),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def _spec(
    entry: ProofManifestEntry,
    *,
    evidence_required: bool = True,
    evidence_schema: str = PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
) -> ProofExecutionSpec:
    return ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=evidence_required,
        evidence_schema=evidence_schema,
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


def _minimal_architecture() -> ArchitectureEvidence:
    from scripts.proof.intergrax_platform_proof_evidence import (
        ParticipantClass,
        ParticipantEvidence,
    )

    participant = ParticipantEvidence(
        participant_id="p1",
        name="n",
        implementation="i",
        version_or_model="v",
        role="r",
        participant_class=ParticipantClass.PLATFORM,
    )
    return ArchitectureEvidence(participants=(participant,))


def _minimal_evidence(
    *,
    proof_id: str = _FAKE_PROOF_ID,
    execution_status: ProofEvidenceExecutionStatus = ProofEvidenceExecutionStatus.PASS,
    source_revision: str = "abc123def456",
    evaluator: EvaluatorSummaryEvidence | None = None,
    scenarios: tuple[ScenarioEvidence, ...] = (),
) -> PlatformProofEvidence:
    started = datetime(2026, 8, 22, 8, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 8, 22, 8, 1, 0, tzinfo=UTC)
    return PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=proof_id,
            title="title",
            domain="test",
            proof_version="1.0.0",
            source_revision=source_revision,
            execution_profile=ProofProfile.QUICK,
        ),
        execution=ExecutionMetadataEvidence(
            status=execution_status,
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
        architecture=_minimal_architecture(),
        participants=_minimal_architecture().participants,
        environment=EnvironmentEvidence(),
        scenarios=scenarios,
        reproduction=ReproductionEvidence(
            source_revision=source_revision,
            command="python run_proof.py",
        ),
        provenance=ProvenanceEvidence(
            proof_id=proof_id,
            source_revision=source_revision,
            generated_at=finished,
            execution_id="exec-1",
            artifact_identity="artifact-1",
        ),
        evaluator=evaluator,
    )


def _write_evidence(
    tmp_path: Path,
    evidence: PlatformProofEvidence,
    *,
    proof_id: str = _FAKE_PROOF_ID,
) -> tuple[Path, Path]:
    artifact_dir = tmp_path / "proof-artifacts" / proof_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = write_evidence_json(evidence, proof_directory=artifact_dir)
    return artifact_dir, path


def _verify(
    artifact_dir: Path,
    evidence_path: Path,
    transport: ProofRunResult,
    *,
    source_revision: str = "abc123def456",
    spec: ProofExecutionSpec | None = None,
) -> EvidenceVerificationResult:
    entry = _entry()
    return verify_platform_proof_evidence(
        evidence_path=evidence_path,
        artifact_root=artifact_dir,
        spec=spec or _spec(entry),
        subprocess_result=transport,
        expected_source_revision=source_revision,
    )


def test_exit_zero_valid_pass_evidence_passes(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(tmp_path, _minimal_evidence())
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.PASS
    final = apply_evidence_verification(_transport(), result)
    assert final.status == ProofStatus.PASS
    assert final.diagnostic_summary == "evidence_verified"


def test_exit_zero_missing_evidence_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "empty"
    artifact_dir.mkdir()
    evidence_path = artifact_dir / EVIDENCE_FILENAME
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.MISSING
    final = apply_evidence_verification(_transport(), result)
    assert final.status == ProofStatus.FAIL
    assert final.diagnostic_summary == "required evidence.json is missing"


def test_exit_zero_malformed_json_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bad-json"
    artifact_dir.mkdir()
    evidence_path = artifact_dir / EVIDENCE_FILENAME
    evidence_path.write_text("{not json", encoding="utf-8")
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.INVALID


def test_exit_zero_invalid_platform_evidence_fails(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "invalid-model"
    artifact_dir.mkdir()
    evidence_path = artifact_dir / EVIDENCE_FILENAME
    evidence_path.write_text(json.dumps({"schema_version": "x"}), encoding="utf-8")
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.INVALID


def test_wrong_schema_version_fails(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(tmp_path, _minimal_evidence())
    entry = _entry()
    result = _verify(
        artifact_dir,
        evidence_path,
        _transport(),
        spec=_spec(entry, evidence_schema="intergrax.platform_proof_evidence.v2"),
    )
    assert result.status == EvidenceVerificationStatus.INVALID


def test_wrong_proof_identity_id_fails(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(proof_id="OTHER-PROOF"),
    )
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.INVALID


def test_wrong_provenance_proof_id_fails(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    provenance = evidence.provenance.model_copy(update={"proof_id": "OTHER-PROOF"})
    evidence = evidence.model_copy(update={"provenance": provenance})
    artifact_dir, evidence_path = _write_evidence(tmp_path, evidence)
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.INVALID


def test_source_revision_mismatch_fails(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(source_revision="other-sha"),
    )
    result = _verify(
        artifact_dir,
        evidence_path,
        _transport(),
        source_revision="abc123def456",
    )
    assert result.status == EvidenceVerificationStatus.INVALID


def test_exit_zero_evidence_fail_status_fails(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(execution_status=ProofEvidenceExecutionStatus.FAIL),
    )
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.FAIL
    assert result.diagnostic_code == "evidence_status_mismatch"


def test_exit_nonzero_evidence_pass_fails(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(tmp_path, _minimal_evidence())
    transport = _transport(status=ProofStatus.FAIL, exit_code=1, diagnostic="child_exit_nonzero")
    result = _verify(artifact_dir, evidence_path, transport)
    assert result.status == EvidenceVerificationStatus.FAIL
    assert result.diagnostic_code == "transport_evidence_mismatch"


def test_exit_nonzero_evidence_fail_preserves_child_diagnostic(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(execution_status=ProofEvidenceExecutionStatus.FAIL),
    )
    transport = _transport(status=ProofStatus.FAIL, exit_code=1, diagnostic="child_exit_nonzero")
    result = _verify(artifact_dir, evidence_path, transport)
    assert result.status == EvidenceVerificationStatus.PASS
    final = apply_evidence_verification(transport, result)
    assert final.status == ProofStatus.FAIL
    assert final.diagnostic_summary == "child_exit_nonzero"


def test_pass_evidence_evaluator_false_fails(tmp_path: Path) -> None:
    evaluator = EvaluatorSummaryEvidence(
        passed=False,
        checks=(
            EvaluatorCheckEvidence(
                check_id="c1",
                label="l",
                passed=False,
            ),
        ),
    )
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(evaluator=evaluator),
    )
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.FAIL
    assert result.diagnostic_code == "evaluator_passed_false"


def test_pass_evidence_scenario_evaluator_false_fails(tmp_path: Path) -> None:
    scenario = ScenarioEvidence(
        scenario_id="s1",
        title="t",
        question="q",
        expected_behavior="e",
        falsification_condition="f",
        execution_status=ProofEvidenceExecutionStatus.PASS,
        evaluator=EvaluatorSummaryEvidence(
            passed=False,
            checks=(
                EvaluatorCheckEvidence(
                    check_id="c1",
                    label="l",
                    passed=False,
                ),
            ),
        ),
    )
    artifact_dir, evidence_path = _write_evidence(
        tmp_path,
        _minimal_evidence(scenarios=(scenario,)),
    )
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.FAIL
    assert result.diagnostic_code.startswith("scenario_evaluator_false")


def test_pass_evidence_without_evaluator_allowed(tmp_path: Path) -> None:
    artifact_dir, evidence_path = _write_evidence(tmp_path, _minimal_evidence())
    result = _verify(artifact_dir, evidence_path, _transport())
    assert result.status == EvidenceVerificationStatus.PASS


def test_legacy_proof_without_evidence_policy_uses_exit_code(tmp_path: Path) -> None:
    entry = ProofManifestEntry(
        proof_id="LEGACY",
        title="legacy",
        domain="test",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="legacy",
        command=ProofArgvCommand(executable="python", argv=("-c", "import sys; sys.exit(0)")),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )
    result = execute_proof(entry, repo_root=tmp_path)
    assert result.status == ProofStatus.PASS
    assert result.evidence_verification_status is None


def test_evidence_path_outside_artifact_root_rejected(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "proof-artifacts"
    artifact_dir.mkdir()
    outside = tmp_path / "outside-evidence.json"
    artifact_dir_evidence, _ = _write_evidence(tmp_path, _minimal_evidence())
    outside.write_text(
        (artifact_dir_evidence / EVIDENCE_FILENAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = _verify(artifact_dir, outside, _transport())
    assert result.status == EvidenceVerificationStatus.INVALID
    assert result.diagnostic_code == "evidence_path_outside_artifact_root"


def test_suite_receipt_does_not_embed_full_evidence(tmp_path: Path) -> None:
    evidence = _minimal_evidence()
    artifact_dir, _ = _write_evidence(tmp_path, evidence)
    transport = _transport()
    verification = _verify(artifact_dir, artifact_dir / EVIDENCE_FILENAME, transport)
    final = apply_evidence_verification(transport, verification)
    receipt = SuiteReceipt(
        suite_run_id="suite-1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        git_commit_sha="abc123",
        git_dirty=False,
        profile=ProofProfile.QUICK,
        platform="linux",
        python_version="3.12",
        overall_status="PASS",
        results=(final,),
        passed_count=1,
        failed_count=0,
        blocked_count=0,
        skipped_count=0,
    )
    serialized = json.dumps(receipt.model_dump(mode="json"))
    assert "proof_identity" not in serialized
    assert evidence.claim.claim not in serialized


def _descriptor_payload(
    *,
    proof_id: str,
    entrypoint: str,
    evidence_required: bool = True,
    run_mode: str = "pass",
) -> dict[str, object]:
    return {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "proof_id": proof_id,
        "title": proof_id,
        "domain": "test_evidence_verify",
        "proof_kind": "evidence_verify",
        "package_version": "1.0.0",
        "profiles": ["quick"],
        "command": {
            "executable": "uv",
            "argv": ["run", "python", entrypoint],
        },
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
        "evidence_required": evidence_required,
        "evidence_schema": PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
        "expected_artifacts": [
            {
                "kind": "EVIDENCE_JSON",
                "relative_path": "evidence.json",
                "required": True,
            }
        ],
        "tags": ["run_mode", run_mode],
    }


def _write_fake_proof_package(
    repo_root: Path,
    *,
    proof_id: str = _FAKE_PROOF_ID,
    run_mode: str = "pass",
    evidence_required: bool = True,
) -> Path:
    package = repo_root / _FAKE_DOMAIN_ROOT / "fake_proof"
    package.mkdir(parents=True, exist_ok=True)
    entrypoint = f"platform_proofs/test_evidence_verify/fake_proof/run_proof.py"
    run_proof = package / "run_proof.py"
    run_proof.write_text(
        _fake_run_proof_source(run_mode=run_mode),
        encoding="utf-8",
    )
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            _descriptor_payload(
                proof_id=proof_id,
                entrypoint=entrypoint,
                evidence_required=evidence_required,
                run_mode=run_mode,
            )
        ),
        encoding="utf-8",
    )
    return package


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
            domain="test",
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
    if RUN_MODE == "pass":
        status = ProofEvidenceExecutionStatus.PASS
        exit_code = 0
    elif RUN_MODE == "fail_evidence":
        status = ProofEvidenceExecutionStatus.FAIL
        exit_code = 0
    else:
        status = ProofEvidenceExecutionStatus.PASS
        exit_code = 1
    write_evidence_json(_build_evidence(status), proof_directory=directory)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
'''


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


def test_runner_integration_valid_evidence_passes(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    _use_unknown_git_revision(monkeypatch)

    config = RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root)
    receipt, receipt_path = run_suite(config)

    result = receipt.results[0]
    assert result.status == ProofStatus.PASS
    assert result.diagnostic_summary == "evidence_verified"
    assert result.evidence_verification_status == ContractEvidenceStatus.PASS
    assert receipt_path is not None
    expected_dir = proof_run_artifact_directory(
        repo_root,
        receipt.suite_run_id,
        _FAKE_PROOF_ID,
    )
    assert Path(result.evidence_path) == expected_dir / EVIDENCE_FILENAME
    assert (expected_dir / EVIDENCE_FILENAME).is_file()
    _cleanup_fake_packages(repo_root)


def test_runner_integration_exit_one_with_pass_evidence_fails(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="exit_one")
    _use_unknown_git_revision(monkeypatch)

    receipt, _ = run_suite(RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root))
    result = receipt.results[0]
    assert result.status == ProofStatus.FAIL
    assert result.diagnostic_summary == "transport_evidence_mismatch"
    _cleanup_fake_packages(repo_root)


def test_runner_integration_exit_zero_fail_evidence_fails(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="fail_evidence")
    _use_unknown_git_revision(monkeypatch)

    receipt, _ = run_suite(
        RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root)
    )
    result = receipt.results[0]
    assert result.status == ProofStatus.FAIL
    assert result.diagnostic_summary == "evidence_status_mismatch"
    _cleanup_fake_packages(repo_root)


def test_runner_sets_artifact_directory_env(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_fake_proof_discovery(repo_root, monkeypatch, run_mode="pass")
    _use_unknown_git_revision(monkeypatch)
    observed: list[str] = []

    def _runner(command, **kwargs):
        observed.append(kwargs["env"][INTERGRAX_PROOF_ARTIFACT_DIR_ENV])
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    receipt, _ = run_suite(
        RunnerConfig(profile=ProofProfile.QUICK, repo_root=repo_root),
        subprocess_runner=_runner,
    )
    expected = proof_run_artifact_directory(
        repo_root,
        receipt.suite_run_id,
        _FAKE_PROOF_ID,
    )
    assert observed == [str(expected)]
    _cleanup_fake_packages(repo_root)


def test_tools_standalone_artifact_directory_compat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from platform_proofs.tools.iterative_sql_investigation.artifacts import (
        allocate_run_directory,
        resolve_runner_artifact_directory,
    )

    standalone = allocate_run_directory(
        artifact_root=tmp_path / "standalone",
        run_id="run-1",
    )
    assert standalone == (tmp_path / "standalone" / "run-1").resolve()

    runner_dir = tmp_path / "runner-proof-dir"
    monkeypatch.setenv(INTERGRAX_PROOF_ARTIFACT_DIR_ENV, str(runner_dir))
    assert resolve_runner_artifact_directory() == runner_dir.resolve()
    assert allocate_run_directory() == runner_dir.resolve()


def test_resolve_expected_evidence_path_from_descriptor(tmp_path: Path) -> None:
    from scripts.proof.intergrax_platform_proof_descriptor import (
        ExpectedArtifactKind,
        ExpectedProofArtifact,
    )

    entry = _entry()
    spec = ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=True,
        expected_artifacts=(
            ExpectedProofArtifact(
                kind=ExpectedArtifactKind.EVIDENCE_JSON,
                relative_path="evidence.json",
            ),
        ),
    )
    artifact_dir = tmp_path / "artifacts"
    assert resolve_expected_evidence_path(artifact_dir, spec) == artifact_dir / "evidence.json"
