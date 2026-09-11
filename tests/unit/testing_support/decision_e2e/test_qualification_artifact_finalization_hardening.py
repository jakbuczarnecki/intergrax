# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-L1.R4.R1.A artifact finalization hardening tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.artifact_finalization_contract import (
    QualificationArtifactFinalizationContract,
)
from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    assess_run_attempt_evidence_status,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    AttemptEvidenceStatus,
    CanonicalRunRecord,
    QualificationExperimentIdentity,
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    SourceFingerprintSnapshot,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.derived_artifact_generator import (
    LocalModelProfileError,
    build_local_model_profile,
    generate_derived_artifacts,
)
from testing_support.decision_e2e.local_qualification_session.finalization import (
    FinalizationContext,
    finalize_with_failure_capture,
)
from testing_support.decision_e2e.local_qualification_session.qualification_analysis_builder import (
    build_qualification_analysis,
)
from testing_support.decision_e2e.local_qualification_session.session import LocalQualificationSession
from testing_support.decision_e2e.local_qualification_session.versioning import (
    ProviderRuntimeVersion,
)


def _identity(**overrides: object) -> QualificationExperimentIdentity:
    base = QualificationExperimentIdentity(
        provider_kind="ollama",
        provider_runtime_version=ProviderRuntimeVersion(0, 34, 0),
        provider_runtime_version_policy=VersionMatchPolicy.EXACT,
        model_name="qwen2.5:14b",
        model_digest="digest-a",
        model_digest_policy=VersionMatchPolicy.EXACT,
        quantization=None,
        generation_config_fingerprint="gen",
        scenario_id="ai_incident_investigation",
        input_id="canonical",
        source_fingerprint="source-a",
        config_fingerprint="config-a",
    )
    return base if not overrides else QualificationExperimentIdentity(**{**base.__dict__, **overrides})


def _observed() -> QualificationRuntimeIdentity:
    return QualificationRuntimeIdentity(
        provider_kind="ollama",
        runtime_version=ProviderRuntimeVersion(0, 34, 0),
        endpoint_host="http://127.0.0.1:11434",
        model_name="qwen2.5:14b",
        model_digest="digest-a",
        quantization=None,
    )


def _session(tmp_path: Path, *, run_count: int = 2) -> LocalQualificationSession:
    frozen = SourceFingerprintSnapshot(repository_head_sha="head", blobs=())
    spec = QualificationSpec(
        experiment_identity=_identity(),
        run_count=run_count,
        required_artifacts=DEFAULT_REQUIRED_ARTIFACTS,
        source_blob_paths=(),
        semantic_source_groups={},
        max_evaluator_attempt_index=1,
        source_checkpoint_run_indices=(0,),
    )
    return LocalQualificationSession(
        session_dir=tmp_path,
        spec=spec,
        frozen_source=frozen,
        session_id="sess",
        task_id="task",
    )


def _seed_runs(session: LocalQualificationSession, tmp_path: Path) -> None:
    session.start(
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    session.persist_canonical_run(CanonicalRunRecord(0, "run-0", ()))
    session.persist_canonical_run(CanonicalRunRecord(1, "run-1", ()))
    (tmp_path / "run.log").write_text("run_index=0 run_id=run-0\n", encoding="utf-8")


def test_missing_required_artifact_blocks_finalize(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _seed_runs(session, tmp_path)
    (tmp_path / "run.log").unlink()
    integrity = session.integrity_report(_observed(), config_matches=True, source_matches=True)
    finalized = session.finalize(integrity=integrity, observed=_observed())
    assert finalized.finalization_status is QualificationSessionState.FAILED_ARTIFACT_VALIDATION


def test_finalize_only_generates_missing_derived_artifacts(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _seed_runs(session, tmp_path)
    runs_payload = {"runs": [{"run_id": "run-0"}, {"run_id": "run-1"}]}
    analysis = build_qualification_analysis(runs_payload, session.spec)
    generate_derived_artifacts(
        tmp_path,
        spec=session.spec,
        analysis=analysis,
        runs_payload=runs_payload,
        observed=_observed(),
        temperature=0.0,
        overwrite=False,
    )
    (tmp_path / "failure_cohort.csv").unlink()
    integrity = session.integrity_report(_observed(), config_matches=True, source_matches=True)
    finalized = session.finalize(
        integrity=integrity,
        observed=_observed(),
        regenerate_derived=True,
    )
    assert finalized.finalization_status is QualificationSessionState.FINALIZED
    assert (tmp_path / "failure_cohort.csv").is_file()


def test_manifest_detects_checksum_mismatch(tmp_path: Path) -> None:
    session = _session(tmp_path, run_count=1)
    session.start(_observed(), config_fingerprint="config-a", source_fingerprint="source-a")
    session.persist_canonical_run(CanonicalRunRecord(0, "run-0", ()))
    (tmp_path / "run.log").write_text("run_index=0\n", encoding="utf-8")
    integrity = session.integrity_report(_observed(), config_matches=True, source_matches=True)
    session.finalize(integrity=integrity, observed=_observed())
    manifest = tmp_path / "artifact-manifest.txt"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("runs.json", "runs.json tampered"), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        QualificationArtifactFinalizationContract.validate_manifest_checksums(tmp_path)


def test_resume_is_idempotent(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _seed_runs(session, tmp_path)
    integrity = session.integrity_report(_observed(), config_matches=True, source_matches=True)
    first = session.finalize(integrity=integrity, observed=_observed())
    assert first.finalization_status is QualificationSessionState.FINALIZED
    cohort_first = (tmp_path / "failure_cohort.csv").read_text(encoding="utf-8")
    session.prepare_finalize_only()
    second = session.finalize(integrity=integrity, observed=_observed(), regenerate_derived=True)
    assert second.finalization_status is QualificationSessionState.FINALIZED
    cohort_second = (tmp_path / "failure_cohort.csv").read_text(encoding="utf-8")
    assert cohort_first == cohort_second
    runs_payload = (tmp_path / "runs.json").read_text(encoding="utf-8")
    assert '"run_id": "run-0"' in runs_payload


def test_model_profile_requires_identity() -> None:
    with pytest.raises(LocalModelProfileError):
        build_local_model_profile(
            _identity(),
            QualificationRuntimeIdentity(
                provider_kind="ollama",
                runtime_version=None,
                endpoint_host="http://127.0.0.1:11434",
                model_name="qwen2.5:14b",
                model_digest="digest-a",
                quantization=None,
            ),
            temperature=0.0,
        )


def test_attempt_evidence_status_is_explicit() -> None:
    assert assess_run_attempt_evidence_status(()) is AttemptEvidenceStatus.NOT_REQUIRED
    from testing_support.decision_e2e.local_qualification_session.contracts import (
        CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    )

    events = (
        {
            "payload_schema_id": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
            "payload": {"run_id": "r", "node_id": "n", "attempt_index": 0},
        },
    )
    assert assess_run_attempt_evidence_status(events) is AttemptEvidenceStatus.AVAILABLE


def test_inject_failure_marks_generation_failed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _seed_runs(session, tmp_path)
    integrity = session.integrity_report(_observed(), config_matches=True, source_matches=True)

    def _boom() -> None:
        raise RuntimeError("derived report failed")

    outcome = finalize_with_failure_capture(
        FinalizationContext(
            session_dir=tmp_path,
            spec=session.spec,
            session_id="sess",
            task_id="task",
            observed=_observed(),
        ),
        integrity=integrity,
        inject_derived_failure=_boom,
    )
    assert outcome.state is QualificationSessionState.FAILED_ARTIFACT_GENERATION
