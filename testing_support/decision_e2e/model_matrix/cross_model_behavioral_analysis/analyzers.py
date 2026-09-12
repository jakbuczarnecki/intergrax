# © Artur Czarnecki. All rights reserved.

"""Built-in behavior analyzers (extend via new classes, not core conditionals)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    AreaAnalysisStatus,
    CohortExecutionDifference,
    ComparisonArea,
    ComparisonAreaFinding,
    PairwiseBehavioralDifference,
    QualificationExitDifference,
    SessionLifecycleDifference,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)


def _pairwise_differences(
    outcomes: tuple[ModelQualificationOutcome, ...],
    *,
    comparison_area: ComparisonArea,
    analyzer_id: str,
    value_fn,
    difference_factory,
) -> tuple[PairwiseBehavioralDifference, ...]:
    differences: list[PairwiseBehavioralDifference] = []
    for left_index in range(len(outcomes)):
        for right_index in range(left_index + 1, len(outcomes)):
            left = outcomes[left_index]
            right = outcomes[right_index]
            left_value = value_fn(left)
            right_value = value_fn(right)
            if left_value == right_value:
                continue
            differences.append(
                difference_factory(
                    left=left,
                    right=right,
                    left_value=left_value,
                    right_value=right_value,
                    comparison_area=comparison_area,
                    analyzer_id=analyzer_id,
                )
            )
    return tuple(differences)


class CohortExecutionBehaviorAnalyzer:
    analyzer_id = "cohort_execution_behavior"
    comparison_area = ComparisonArea.COHORT_EXECUTION

    def analyze(
        self,
        outcomes: tuple[ModelQualificationOutcome, ...],
    ) -> ComparisonAreaFinding:
        if len(outcomes) < 2:
            return ComparisonAreaFinding(
                comparison_area=self.comparison_area,
                analyzer_id=self.analyzer_id,
                status=AreaAnalysisStatus.INSUFFICIENT_DATA,
                differences=(),
            )
        differences = _pairwise_differences(
            outcomes,
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            value_fn=lambda item: item.status,
            difference_factory=_cohort_execution_difference,
        )
        status = (
            AreaAnalysisStatus.UNIFORM
            if not differences
            else AreaAnalysisStatus.DIFFERS
        )
        return ComparisonAreaFinding(
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            status=status,
            differences=differences,
        )


def _cohort_execution_difference(
    *,
    left: ModelQualificationOutcome,
    right: ModelQualificationOutcome,
    left_value,
    right_value,
    comparison_area: ComparisonArea,
    analyzer_id: str,
) -> CohortExecutionDifference:
    return CohortExecutionDifference(
        comparison_area=comparison_area,
        analyzer_id=analyzer_id,
        profile_key_left=left.profile_key,
        profile_key_right=right.profile_key,
        left_descriptor=str(left_value),
        right_descriptor=str(right_value),
        left_status=left_value,
        right_status=right_value,
    )


class QualificationExitBehaviorAnalyzer:
    analyzer_id = "qualification_exit_behavior"
    comparison_area = ComparisonArea.QUALIFICATION_EXIT

    def analyze(
        self,
        outcomes: tuple[ModelQualificationOutcome, ...],
    ) -> ComparisonAreaFinding:
        if len(outcomes) < 2:
            return ComparisonAreaFinding(
                comparison_area=self.comparison_area,
                analyzer_id=self.analyzer_id,
                status=AreaAnalysisStatus.INSUFFICIENT_DATA,
                differences=(),
            )
        differences = _pairwise_differences(
            outcomes,
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            value_fn=lambda item: item.exit_code,
            difference_factory=_qualification_exit_difference,
        )
        status = (
            AreaAnalysisStatus.UNIFORM
            if not differences
            else AreaAnalysisStatus.DIFFERS
        )
        return ComparisonAreaFinding(
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            status=status,
            differences=differences,
        )


def _qualification_exit_difference(
    *,
    left: ModelQualificationOutcome,
    right: ModelQualificationOutcome,
    left_value,
    right_value,
    comparison_area: ComparisonArea,
    analyzer_id: str,
) -> QualificationExitDifference:
    return QualificationExitDifference(
        comparison_area=comparison_area,
        analyzer_id=analyzer_id,
        profile_key_left=left.profile_key,
        profile_key_right=right.profile_key,
        left_descriptor=left_value.name,
        right_descriptor=right_value.name,
        left_exit_code=left_value,
        right_exit_code=right_value,
    )


class SessionLifecycleBehaviorAnalyzer:
    analyzer_id = "session_lifecycle_behavior"
    comparison_area = ComparisonArea.SESSION_LIFECYCLE

    def analyze(
        self,
        outcomes: tuple[ModelQualificationOutcome, ...],
    ) -> ComparisonAreaFinding:
        if len(outcomes) < 2:
            return ComparisonAreaFinding(
                comparison_area=self.comparison_area,
                analyzer_id=self.analyzer_id,
                status=AreaAnalysisStatus.INSUFFICIENT_DATA,
                differences=(),
            )
        differences = _pairwise_differences(
            outcomes,
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            value_fn=lambda item: item.session_state,
            difference_factory=_session_lifecycle_difference,
        )
        status = (
            AreaAnalysisStatus.UNIFORM
            if not differences
            else AreaAnalysisStatus.DIFFERS
        )
        return ComparisonAreaFinding(
            comparison_area=self.comparison_area,
            analyzer_id=self.analyzer_id,
            status=status,
            differences=differences,
        )


def _session_lifecycle_difference(
    *,
    left: ModelQualificationOutcome,
    right: ModelQualificationOutcome,
    left_value,
    right_value,
    comparison_area: ComparisonArea,
    analyzer_id: str,
) -> SessionLifecycleDifference:
    left_label = left_value.value if left_value is not None else "none"
    right_label = right_value.value if right_value is not None else "none"
    return SessionLifecycleDifference(
        comparison_area=comparison_area,
        analyzer_id=analyzer_id,
        profile_key_left=left.profile_key,
        profile_key_right=right.profile_key,
        left_descriptor=left_label,
        right_descriptor=right_label,
        left_session_state=left_value,
        right_session_state=right_value,
    )


def default_behavior_analyzers() -> tuple[
    CohortExecutionBehaviorAnalyzer,
    QualificationExitBehaviorAnalyzer,
    SessionLifecycleBehaviorAnalyzer,
]:
    return (
        CohortExecutionBehaviorAnalyzer(),
        QualificationExitBehaviorAnalyzer(),
        SessionLifecycleBehaviorAnalyzer(),
    )


__all__ = [
    "CohortExecutionBehaviorAnalyzer",
    "QualificationExitBehaviorAnalyzer",
    "SessionLifecycleBehaviorAnalyzer",
    "default_behavior_analyzers",
]
