# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis import (
    CrossModelBehavioralAnalysisRequest,
    run_cross_model_behavioral_analysis,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline import (
    BASELINE_TASK_ID,
    BASELINE_VERSION,
    CapabilityDimensionId,
    CapabilityProfileBuildRequest,
    CapabilityProfileBuildStatus,
    ObservationLevel,
    build_model_capability_profiles,
    default_capability_extractors,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
)


def _outcome(
    profile_key: str,
    *,
    exit_code: QualificationCliExit = QualificationCliExit.SUCCESS,
    status: CohortExecutionStatus = CohortExecutionStatus.EXECUTED,
    session_state: QualificationSessionState
    | None = QualificationSessionState.FINALIZED,
    model_name: str | None = None,
    matrix_version: str | None = None,
) -> ModelQualificationOutcome:
    stamp = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="ollama",
        model_name=model_name or profile_key,
        matrix_version=matrix_version or qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=status,
        exit_code=exit_code,
        session_state=session_state,
    )


def test_single_model_generates_capability_profile() -> None:
    matrix_version = qualification_matrix_version()
    request = CapabilityProfileBuildRequest(
        matrix_version=matrix_version,
        outcomes=(_outcome("model-a"),),
    )
    result = build_model_capability_profiles(
        request,
        generated_at=datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC),
    )

    assert result.status is CapabilityProfileBuildStatus.COMPLETE
    assert result.build_task_id == BASELINE_TASK_ID
    assert result.baseline_version == BASELINE_VERSION
    assert len(result.profiles) == 1
    profile = result.profiles[0]
    assert profile.model_identity.profile_key == "model-a"
    assert len(profile.qualification_source_refs) == 1
    assert (
        profile.qualification_source_refs[0].qualification_task_id == "DS-E2E-15J-L1.R6"
    )
    exit_caps = [
        item
        for item in profile.capabilities
        if item.dimension_id is CapabilityDimensionId.QUALIFICATION_EXIT
    ]
    assert len(exit_caps) == 1
    assert exit_caps[0].level is ObservationLevel.STRONG


def test_two_models_receive_separate_profiles() -> None:
    matrix_version = qualification_matrix_version()
    request = CapabilityProfileBuildRequest(
        matrix_version=matrix_version,
        outcomes=(
            _outcome("model-a"),
            _outcome(
                "model-b",
                exit_code=QualificationCliExit.CRITICAL_SAFETY_FAILURE,
            ),
        ),
    )
    result = build_model_capability_profiles(request)

    assert result.status is CapabilityProfileBuildStatus.COMPLETE
    assert len(result.profiles) == 2
    keys = {item.model_identity.profile_key for item in result.profiles}
    assert keys == {"model-a", "model-b"}

    profile_b = next(
        item for item in result.profiles if item.model_identity.profile_key == "model-b"
    )
    exit_limits = [
        item
        for item in profile_b.limitations
        if item.dimension_id is CapabilityDimensionId.QUALIFICATION_EXIT
    ]
    assert len(exit_limits) == 1
    assert exit_limits[0].level is ObservationLevel.WEAK


def test_custom_extractor_plugs_in_without_engine_edit() -> None:
    class _StubDimensionExtractor:
        extractor_id = "stub_dimension"
        dimension_id = CapabilityDimensionId.SESSION_LIFECYCLE

        def extract(self, *, profile_key, outcome, behavioral_comparison):
            _ = profile_key, behavioral_comparison
            from testing_support.decision_e2e.model_matrix.model_capability_baseline.protocol import (
                CapabilityExtractorResult,
            )
            from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
                CapabilityObservation,
            )
            from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
                OutcomeSourceRef,
            )

            if outcome is None:
                return CapabilityExtractorResult(capabilities=(), limitations=())
            ref = OutcomeSourceRef(
                profile_key=outcome.profile_key,
                qualification_task_id=outcome.qualification_task_id,
                matrix_version=outcome.matrix_version,
                evaluated_at=outcome.evaluated_at,
            )
            return CapabilityExtractorResult(
                capabilities=(
                    CapabilityObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.NEUTRAL,
                        factual_descriptor="stub extractor observation",
                        qualification_evidence=(ref,),
                        behavioral_evidence=(),
                    ),
                ),
                limitations=(),
            )

    matrix_version = qualification_matrix_version()
    request = CapabilityProfileBuildRequest(
        matrix_version=matrix_version,
        outcomes=(_outcome("model-a"),),
    )
    builtins = default_capability_extractors()
    result = build_model_capability_profiles(
        request,
        extractors=(*builtins, _StubDimensionExtractor()),
    )
    profile = result.profiles[0]
    stub_obs = [
        item for item in profile.capabilities if item.extractor_id == "stub_dimension"
    ]
    assert len(stub_obs) == 1


def test_insufficient_data_empty_outcomes() -> None:
    request = CapabilityProfileBuildRequest(
        matrix_version=qualification_matrix_version(),
        outcomes=(),
    )
    result = build_model_capability_profiles(request)
    assert result.status is CapabilityProfileBuildStatus.INSUFFICIENT_DATA
    assert result.profiles == ()


def test_behavioral_comparison_refs_integrated() -> None:
    matrix_version = qualification_matrix_version()
    outcomes = (
        _outcome("model-a"),
        _outcome("model-b", exit_code=QualificationCliExit.CRITICAL_SAFETY_FAILURE),
    )
    behavioral = run_cross_model_behavioral_analysis(
        CrossModelBehavioralAnalysisRequest(
            scenario_id="cap-baseline",
            matrix_version=matrix_version,
            outcomes=outcomes,
        )
    )
    result = build_model_capability_profiles(
        CapabilityProfileBuildRequest(
            matrix_version=matrix_version,
            outcomes=outcomes,
            behavioral_comparison=behavioral,
        )
    )
    profile_b = next(
        item for item in result.profiles if item.model_identity.profile_key == "model-b"
    )
    assert profile_b.behavioral_source_ref is not None
    peer_limits = [
        item
        for item in profile_b.limitations
        if item.dimension_id is CapabilityDimensionId.BEHAVIORAL_PEER_ALIGNMENT
    ]
    assert peer_limits
