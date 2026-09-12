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
    ANALYSIS_TASK_ID,
    AreaAnalysisStatus,
    ComparisonArea,
    CrossModelAnalysisStatus,
    CrossModelBehavioralAnalysisRequest,
    QualificationExitBehaviorAnalyzer,
    default_behavior_analyzers,
    run_cross_model_behavioral_analysis,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    QualificationExitDifference,
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
    stamp = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)
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


def test_basic_two_model_comparison_detects_exit_difference() -> None:
    matrix_version = qualification_matrix_version()
    request = CrossModelBehavioralAnalysisRequest(
        scenario_id="scenario-alpha",
        matrix_version=matrix_version,
        outcomes=(
            _outcome("model-a"),
            _outcome("model-b", exit_code=QualificationCliExit.CRITICAL_SAFETY_FAILURE),
        ),
    )
    result = run_cross_model_behavioral_analysis(request)

    assert result.status is CrossModelAnalysisStatus.COMPLETE
    assert result.analysis_task_id == ANALYSIS_TASK_ID
    assert len(result.models) == 2
    assert len(result.source_outcome_refs) == 2
    assert result.scenario_id == "scenario-alpha"

    exit_finding = next(
        item
        for item in result.findings
        if item.comparison_area is ComparisonArea.QUALIFICATION_EXIT
    )
    assert exit_finding.status is AreaAnalysisStatus.DIFFERS
    assert len(exit_finding.differences) == 1
    diff = exit_finding.differences[0]
    assert isinstance(diff, QualificationExitDifference)
    assert diff.profile_key_left == "model-a"
    assert diff.profile_key_right == "model-b"


def test_three_models_without_core_changes() -> None:
    matrix_version = qualification_matrix_version()
    request = CrossModelBehavioralAnalysisRequest(
        scenario_id="scenario-beta",
        matrix_version=matrix_version,
        outcomes=(
            _outcome("model-a"),
            _outcome("model-b"),
            _outcome("model-c"),
        ),
    )
    result = run_cross_model_behavioral_analysis(request)

    assert result.status is CrossModelAnalysisStatus.COMPLETE
    assert len(result.models) == 3
    assert all(item.status is AreaAnalysisStatus.UNIFORM for item in result.findings)


def test_insufficient_data_empty_outcomes() -> None:
    request = CrossModelBehavioralAnalysisRequest(
        scenario_id="empty",
        matrix_version=qualification_matrix_version(),
        outcomes=(),
    )
    result = run_cross_model_behavioral_analysis(request)
    assert result.status is CrossModelAnalysisStatus.INSUFFICIENT_DATA
    assert result.findings == ()


def test_version_mismatch_is_controlled() -> None:
    request = CrossModelBehavioralAnalysisRequest(
        scenario_id="mismatch",
        matrix_version=qualification_matrix_version(),
        outcomes=(
            _outcome("model-a"),
            _outcome("model-b", matrix_version="other-version"),
        ),
    )
    result = run_cross_model_behavioral_analysis(request)
    assert result.status is CrossModelAnalysisStatus.VERSION_MISMATCH
    assert result.findings == ()


def test_custom_analyzer_plugs_in_without_engine_edit() -> None:
    class _UniformExitStub:
        analyzer_id = "stub_exit"
        comparison_area = ComparisonArea.QUALIFICATION_EXIT

        def analyze(self, outcomes):
            return QualificationExitBehaviorAnalyzer().analyze(outcomes)

    matrix_version = qualification_matrix_version()
    request = CrossModelBehavioralAnalysisRequest(
        scenario_id="plugin",
        matrix_version=matrix_version,
        outcomes=(_outcome("model-a"), _outcome("model-b")),
    )
    builtins = default_behavior_analyzers()
    result = run_cross_model_behavioral_analysis(
        request,
        analyzers=(*builtins, _UniformExitStub()),
    )
    assert result.status is CrossModelAnalysisStatus.COMPLETE
    assert len(result.findings) == len(builtins) + 1
