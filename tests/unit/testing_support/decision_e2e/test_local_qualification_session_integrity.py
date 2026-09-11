# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-QI1 local qualification session integrity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)
from testing_support.decision_e2e.local_qualification_session.attempt_evidence import (
    assess_third_model_pass,
    extract_attempt_observations,
)
from testing_support.decision_e2e.local_qualification_session.classification_adapter import (
    failure_view_from_classification,
)
from testing_support.decision_e2e.local_qualification_session.artifact_contract import (
    DEFAULT_REQUIRED_ARTIFACTS,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    CanonicalRunRecord,
    QualificationExperimentIdentity,
    QualificationPreconditionFailureKind,
    QualificationRuntimeIdentity,
    QualificationSessionState,
    QualificationSpec,
    RECONCILIATION_PHASE_TRACE_SCHEMA,
    VersionMatchPolicy,
)
from testing_support.decision_e2e.local_qualification_session.finalization import (
    ArtifactValidationError,
    FinalizationContext,
    finalize_with_failure_capture,
    validate_required_artifacts,
)
from testing_support.decision_e2e.local_qualification_session.identity import evaluate_preconditions
from testing_support.decision_e2e.local_qualification_session.reconciliation_leak import (
    assess_reconciliation_leak,
    extract_reconciliation_phase_observations,
)
from testing_support.decision_e2e.local_qualification_session.run_registry import (
    RunRegistry,
    RunRegistryError,
)
from testing_support.decision_e2e.local_qualification_session.session import LocalQualificationSession
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_source_fingerprint,
    compare_source_snapshots,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)
from testing_support.decision_e2e.local_qualification_session.versioning import (
    ProviderRuntimeVersion,
    compare_runtime_versions,
)


def _identity(**overrides: object) -> QualificationExperimentIdentity:
    base = QualificationExperimentIdentity(
        provider_kind="ollama",
        provider_runtime_version=ProviderRuntimeVersion(0, 33, 3),
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


def _observed(
    *,
    version: ProviderRuntimeVersion | None = ProviderRuntimeVersion(0, 33, 3),
    digest: str = "digest-a",
) -> QualificationRuntimeIdentity:
    return QualificationRuntimeIdentity(
        provider_kind="ollama",
        runtime_version=version,
        endpoint_host="http://127.0.0.1:11434",
        model_name="qwen2.5:14b",
        model_digest=digest,
        quantization=None,
    )


def _alignment_event(**fields: object) -> dict[str, object]:
    payload = {
        "alignment_mismatch_detected": False,
        "alignment_direction": None,
        "alignment_correctable": False,
        "alignment_correction_attempted": False,
        "alignment_correction_succeeded": False,
        "alignment_correction_exhausted": False,
        "revision_authoritative_context_present": False,
    }
    payload.update(fields)
    return {
        "payload_schema_id": COMPLETION_ALIGNMENT_TRACE_SCHEMA,
        "payload": payload,
    }


def _attempt_event(run_id: str, attempt_index: int) -> dict[str, object]:
    return {
        "payload_schema_id": CANONICAL_MODEL_ATTEMPT_TRACE_SCHEMA,
        "payload": {
            "run_id": run_id,
            "node_id": "investigator",
            "attempt_index": attempt_index,
        },
    }


def _reconciliation_event(
    *,
    validation_invalid: bool,
    entered_reconciliation: bool,
) -> dict[str, object]:
    return {
        "payload_schema_id": RECONCILIATION_PHASE_TRACE_SCHEMA,
        "payload": {
            "run_id": "run-1",
            "attempt_index": 0,
            "validation_invalid": validation_invalid,
            "entered_reconciliation": entered_reconciliation,
        },
    }


def test_wrong_ollama_version_blocks_before_run() -> None:
    result = evaluate_preconditions(
        _identity(),
        _observed(version=ProviderRuntimeVersion(0, 34, 0)),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        strict_tool_capable=True,
        structured_output_capable=True,
    )
    assert not result.eligible
    assert any(
        item.kind is QualificationPreconditionFailureKind.PROVIDER_RUNTIME_MISMATCH
        for item in result.failures
    )


def test_exact_ollama_version_passes_gate() -> None:
    result = evaluate_preconditions(
        _identity(),
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        strict_tool_capable=True,
        structured_output_capable=True,
    )
    assert result.eligible


def test_model_digest_mismatch_blocks() -> None:
    result = evaluate_preconditions(
        _identity(),
        _observed(digest="digest-b"),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        strict_tool_capable=True,
        structured_output_capable=True,
    )
    assert not result.eligible
    assert any(
        item.kind is QualificationPreconditionFailureKind.MODEL_IDENTITY_MISMATCH
        for item in result.failures
    )


def test_config_drift_blocks() -> None:
    result = evaluate_preconditions(
        _identity(),
        _observed(),
        config_fingerprint="config-b",
        source_fingerprint="source-a",
        strict_tool_capable=True,
        structured_output_capable=True,
    )
    assert not result.eligible


def test_semantic_source_drift_detected(tmp_path: Path) -> None:
    blob = tmp_path / "blob.txt"
    blob.write_text("v1", encoding="utf-8")
    frozen = capture_source_fingerprint(
        tmp_path,
        blob_paths=("blob.txt",),
        semantic_group="g",
        repository_head_sha="aaa",
    )
    blob.write_text("v2", encoding="utf-8")
    current = capture_source_fingerprint(
        tmp_path,
        blob_paths=("blob.txt",),
        semantic_group="g",
        repository_head_sha="bbb",
    )
    report = compare_source_snapshots(frozen, current)
    assert report.repository_head_drift
    assert report.qualification_semantic_source_drift


def test_orthogonal_head_drift_without_semantic_drift(tmp_path: Path) -> None:
    blob = tmp_path / "blob.txt"
    blob.write_text("stable", encoding="utf-8")
    frozen = capture_source_fingerprint(
        tmp_path,
        blob_paths=("blob.txt",),
        semantic_group="g",
        repository_head_sha="aaa",
    )
    current = capture_source_fingerprint(
        tmp_path,
        blob_paths=("blob.txt",),
        semantic_group="g",
        repository_head_sha="bbb",
    )
    report = compare_source_snapshots(frozen, current)
    assert report.repository_head_drift
    assert not report.qualification_semantic_source_drift


def test_trace_readback_empty_success() -> None:
    readback = read_typed_alignment_events(())
    assert readback.status.value == "pass"
    assert readback.events == ()


def test_trace_readback_failure_on_malformed_payload() -> None:
    readback = read_typed_alignment_events(
        (
            {
                "payload_schema_id": COMPLETION_ALIGNMENT_TRACE_SCHEMA,
                "payload": {"alignment_mismatch_detected": "not-bool"},
            },
        )
    )
    assert readback.status.value == "failed"


def test_attempts_zero_one_no_third_pass() -> None:
    events = (_attempt_event("run-1", 0), _attempt_event("run-1", 1))
    observations = extract_attempt_observations(events)
    assessment = assess_third_model_pass(observations, max_valid_attempt_index=1)
    assert assessment.outcome.value == "pass"


def test_attempt_two_is_third_pass() -> None:
    observations = extract_attempt_observations((_attempt_event("run-1", 2),))
    assessment = assess_third_model_pass(observations, max_valid_attempt_index=1)
    assert assessment.outcome.value == "fail"


def test_no_attempt_evidence_is_unknown() -> None:
    assessment = assess_third_model_pass((), max_valid_attempt_index=1)
    assert assessment.outcome.value == "unknown"
    assert not assessment.attempt_evidence_complete


def test_reconciliation_leak_true() -> None:
    observations = extract_reconciliation_phase_observations(
        (_reconciliation_event(validation_invalid=True, entered_reconciliation=True),)
    )
    assessment = assess_reconciliation_leak(observations)
    assert assessment.outcome.value == "fail"


def test_reconciliation_leak_false() -> None:
    observations = extract_reconciliation_phase_observations(
        (_reconciliation_event(validation_invalid=True, entered_reconciliation=False),)
    )
    assessment = assess_reconciliation_leak(observations)
    assert assessment.outcome.value == "pass"


def test_reconciliation_missing_evidence_unknown() -> None:
    assessment = assess_reconciliation_leak(())
    assert assessment.outcome.value == "unknown"


def test_duplicate_run_index_rejected() -> None:
    registry = RunRegistry(2)
    registry.register(CanonicalRunRecord(0, "run-a", ()))
    with pytest.raises(RunRegistryError):
        registry.register(CanonicalRunRecord(0, "run-b", ()))


def test_duplicate_run_id_rejected() -> None:
    registry = RunRegistry(2)
    registry.register(CanonicalRunRecord(0, "run-a", ()))
    with pytest.raises(RunRegistryError):
        registry.register(CanonicalRunRecord(1, "run-a", ()))


def test_out_of_range_run_rejected() -> None:
    registry = RunRegistry(1)
    with pytest.raises(RunRegistryError):
        registry.register(CanonicalRunRecord(3, "run-a", ()))


def _session(tmp_path: Path) -> LocalQualificationSession:
    from testing_support.decision_e2e.local_qualification_session.contracts import (
        SourceFingerprintSnapshot,
    )

    frozen = SourceFingerprintSnapshot(
        repository_head_sha="head",
        blobs=(),
    )
    spec = QualificationSpec(
        experiment_identity=_identity(),
        run_count=2,
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


def test_session_blocked_without_runs(tmp_path: Path) -> None:
    session = _session(tmp_path)
    start = session.start(
        _observed(version=ProviderRuntimeVersion(0, 34, 0)),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    assert start.state is QualificationSessionState.BLOCKED
    assert not (tmp_path / "runs.json").exists()


def test_finalize_crash_sets_failed_finalization(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.start(
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    session.persist_canonical_run(CanonicalRunRecord(0, "run-0", ()))
    session.persist_canonical_run(CanonicalRunRecord(1, "run-1", ()))
    (tmp_path / "run.log").write_text("run_index=0\nrun_index=1\n", encoding="utf-8")
    integrity = session.integrity_report(
        _observed(),
        config_matches=True,
        source_matches=True,
    )

    def _boom() -> None:
        raise RuntimeError("derived report failed")

    outcome = finalize_with_failure_capture(
        FinalizationContext(
            session_dir=tmp_path,
            spec=session._spec,
            session_id="sess",
            task_id="task",
            observed=_observed(),
        ),
        integrity=integrity,
        inject_derived_failure=_boom,
    )
    assert outcome.state is QualificationSessionState.FAILED_ARTIFACT_GENERATION
    assert (tmp_path / "runs.json").is_file()


def test_finalize_recovery_without_model_calls(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.start(
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    session.persist_canonical_run(CanonicalRunRecord(0, "run-0", ()))
    session.persist_canonical_run(CanonicalRunRecord(1, "run-1", ()))
    (tmp_path / "run.log").write_text("run_index=0\nrun_index=1\n", encoding="utf-8")
    integrity = session.integrity_report(
        _observed(),
        config_matches=True,
        source_matches=True,
    )
    finalized = session.finalize(integrity=integrity, observed=_observed())
    assert finalized.finalization_status is QualificationSessionState.FINALIZED
    validate_required_artifacts(tmp_path, session._spec.required_artifacts)


def test_missing_summary_prevents_valid_finalization(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.start(
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    session.persist_canonical_run(CanonicalRunRecord(0, "run-0", ()))
    session.persist_canonical_run(CanonicalRunRecord(1, "run-1", ()))
    (tmp_path / "run.log").write_text("run_index=0\nrun_index=1\n", encoding="utf-8")
    (tmp_path / "summary.json").unlink(missing_ok=True)
    integrity = session.integrity_report(
        _observed(),
        config_matches=True,
        source_matches=True,
    )
    finalized = session.finalize(integrity=integrity, observed=_observed())
    assert finalized.finalization_status is QualificationSessionState.FINALIZED
    assert (tmp_path / "summary.json").is_file()


def test_resume_does_not_rerun_completed_run(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.start(
        _observed(),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    session.persist_canonical_run(
        CanonicalRunRecord(0, "run-0", (_attempt_event("run-0", 0),))
    )
    resumed = LocalQualificationSession(
        session_dir=tmp_path,
        spec=session._spec,
        frozen_source=session._frozen_source,
        session_id="sess",
        task_id="task",
    )
    with pytest.raises(RunRegistryError):
        resumed.persist_canonical_run(CanonicalRunRecord(0, "run-0-replacement", ()))


def test_resume_identity_drift_blocked(tmp_path: Path) -> None:
    from testing_support.decision_e2e.local_qualification_session.contracts import (
        SourceFingerprintSnapshot,
    )

    session = LocalQualificationSession(
        session_dir=tmp_path,
        spec=QualificationSpec(
            experiment_identity=_identity(),
            run_count=1,
            required_artifacts=("runs.json",),
            source_blob_paths=(),
            semantic_source_groups={},
            max_evaluator_attempt_index=1,
            source_checkpoint_run_indices=(0,),
        ),
        frozen_source=SourceFingerprintSnapshot(repository_head_sha="h", blobs=()),
    )
    result = session.assert_resume_identity(
        _observed(version=ProviderRuntimeVersion(0, 34, 0)),
        config_fingerprint="config-a",
        source_fingerprint="source-a",
    )
    assert not result.eligible


def test_classification_adapter_platform_axis() -> None:
    classification = DecisionFailureClassification(
        category=DecisionFailureCategory.PLATFORM_CONTRACT,
        reason=DecisionFailureReason.TRACE_RUN_NOT_FINALIZED,
        boundary=DecisionFailureBoundary.HOST_EXECUTION,
        owner=DecisionFailureOwner.OBSERVABILITY,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_INCOMPLETE_TRACE,
    )
    view = failure_view_from_classification(classification)
    assert view.is_platform_failure


def test_runtime_version_compare_exact() -> None:
    assert (
        compare_runtime_versions(
            ProviderRuntimeVersion(0, 33, 3),
            ProviderRuntimeVersion(0, 33, 3),
            policy=VersionMatchPolicy.EXACT,
        ).value
        == "match"
    )


def test_manifest_mismatch_detected_on_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError):
        validate_required_artifacts(tmp_path, ("summary.json",))
