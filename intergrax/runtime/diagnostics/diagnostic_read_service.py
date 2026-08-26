# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator-facing diagnostic read composition (DIAG-6)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticAssessmentBuilder,
    DiagnosticAssessmentIntegrityError,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticGroupingProvenance,
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemDetail,
    DiagnosticProblemListResult,
    DiagnosticProblemOccurrenceView,
    DiagnosticProblemSummary,
    DiagnosticReadIntegrityError,
    DiagnosticReadUnavailableReason,
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstruction,
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnalysisIntegrityError,
    LifecycleAnomalyAnalyzer,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemOccurrence,
    ProblemStatus,
    validate_problem_id,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence

DEFAULT_PROBLEM_LIST_LIMIT = 100
MAX_PROBLEM_LIST_LIMIT = 1000
DEFAULT_OCCURRENCE_LIMIT = 100
MAX_OCCURRENCE_LIMIT = 1000


class DiagnosticReadService:
    """
    Canonical operator-facing diagnostic read entry point.

    Orchestrates Problem persistence reads and bounded DIAG-2→4 reconstruction
    for occurrence detail. Does not mutate Problems or persist assessments.
    """

    def __init__(
        self,
        problem_persistence: ProblemPersistence,
        execution_reconstructor: ExecutionReconstructor,
        *,
        lifecycle_analyzer: LifecycleAnomalyAnalyzer | None = None,
        assessment_builder: DiagnosticAssessmentBuilder | None = None,
    ) -> None:
        self._persistence = problem_persistence
        self._reconstructor = execution_reconstructor
        self._lifecycle_analyzer = lifecycle_analyzer or LifecycleAnomalyAnalyzer()
        self._assessment_builder = assessment_builder or DiagnosticAssessmentBuilder()

    def list_problems(
        self,
        *,
        tenant_id: str,
        status: ProblemStatus | None = None,
        limit: int = DEFAULT_PROBLEM_LIST_LIMIT,
    ) -> DiagnosticProblemListResult:
        tenant_id = _require_tenant_id(tenant_id)
        limit = _validate_bounded_limit(limit, max_limit=MAX_PROBLEM_LIST_LIMIT)

        records = self._persistence.list_for_tenant(tenant_id)
        if status is not None:
            records = tuple(record for record in records if record.status is status)

        ordered = sorted(
            records,
            key=lambda problem: (-problem.last_seen_at.timestamp(), str(problem.problem_id)),
        )
        total_count = len(ordered)
        selected = ordered[:limit]
        summaries = tuple(_summary_from_problem(problem) for problem in selected)

        return DiagnosticProblemListResult(
            problems=summaries,
            total_count=total_count,
            returned_count=len(summaries),
            is_truncated=total_count > len(summaries),
        )

    def get_problem(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        occurrence_limit: int = DEFAULT_OCCURRENCE_LIMIT,
    ) -> DiagnosticProblemDetail | None:
        tenant_id = _require_tenant_id(tenant_id)
        problem_id = validate_problem_id(problem_id)
        occurrence_limit = _validate_bounded_limit(
            occurrence_limit,
            max_limit=MAX_OCCURRENCE_LIMIT,
        )

        problem = self._persistence.get(tenant_id=tenant_id, problem_id=problem_id)
        if problem is None:
            return None

        if problem.tenant_id != tenant_id:
            raise DiagnosticReadIntegrityError(
                "persisted Problem tenant_id does not match lookup tenant scope",
            )

        grouping_provenance = grouping_provenance_from_problem_provenance(problem.provenance)
        ordered_occurrences = _order_occurrences_newest_first(problem.occurrences)
        total_occurrence_count = len(ordered_occurrences)
        selected_occurrences = ordered_occurrences[:occurrence_limit]

        occurrence_views = tuple(
            _reconstruct_occurrence_view(
                occurrence,
                tenant_id=tenant_id,
                problem=problem,
                reconstructor=self._reconstructor,
                lifecycle_analyzer=self._lifecycle_analyzer,
                assessment_builder=self._assessment_builder,
            )
            for occurrence in selected_occurrences
        )

        return DiagnosticProblemDetail(
            problem_id=problem.problem_id,
            tenant_id=problem.tenant_id,
            status=problem.status,
            first_seen_at=problem.first_seen_at,
            last_seen_at=problem.last_seen_at,
            occurrence_count=problem.occurrence_count,
            record_version=problem.record_version,
            grouping_provenance=grouping_provenance,
            occurrences=occurrence_views,
            returned_occurrence_count=len(occurrence_views),
            total_occurrence_count=total_occurrence_count,
            is_occurrences_truncated=total_occurrence_count > len(occurrence_views),
        )


def _summary_from_problem(problem: Problem) -> DiagnosticProblemSummary:
    return DiagnosticProblemSummary(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=problem.last_seen_at,
        occurrence_count=problem.occurrence_count,
        grouping_provenance=grouping_provenance_from_problem_provenance(problem.provenance),
    )


def _order_occurrences_newest_first(
    occurrences: tuple[ProblemOccurrence, ...],
) -> tuple[ProblemOccurrence, ...]:
    return tuple(
        sorted(
            occurrences,
            key=lambda occurrence: (
                -occurrence.observed_at.timestamp(),
                str(occurrence.subject_ref.task_id),
                str(occurrence.subject_ref.run_id),
            ),
        )
    )


def _reconstruct_occurrence_view(
    occurrence: ProblemOccurrence,
    *,
    tenant_id: str,
    problem: Problem,
    reconstructor: ExecutionReconstructor,
    lifecycle_analyzer: LifecycleAnomalyAnalyzer,
    assessment_builder: DiagnosticAssessmentBuilder,
) -> DiagnosticProblemOccurrenceView:
    subject_ref = occurrence.subject_ref
    if subject_ref.tenant_id != tenant_id:
        raise DiagnosticReadIntegrityError(
            "occurrence subject_ref tenant_id does not match lookup tenant scope",
        )
    if subject_ref.tenant_id != problem.tenant_id:
        raise DiagnosticReadIntegrityError(
            "occurrence subject_ref tenant_id does not match Problem tenant_id",
        )

    try:
        reconstruction = reconstructor.reconstruct_execution(
            tenant_id,
            subject_ref.task_id,
            subject_ref.run_id,
        )
        _validate_reconstruction_scope(reconstruction, subject_ref=subject_ref)
    except ExecutionReconstructionIntegrityError as exc:
        raise DiagnosticReadIntegrityError(str(exc)) from exc

    if _is_execution_evidence_unavailable(reconstruction):
        return DiagnosticProblemOccurrenceView(
            subject_ref=subject_ref,
            observed_at=occurrence.observed_at,
            strategy_id=occurrence.strategy_id,
            strategy_version=occurrence.strategy_version,
            method=occurrence.method,
            read_status=DiagnosticOccurrenceReadStatus.UNAVAILABLE,
            assessment=None,
            unavailable_reason=DiagnosticReadUnavailableReason.EXECUTION_EVIDENCE_UNAVAILABLE,
        )

    try:
        lifecycle = lifecycle_analyzer.analyze(reconstruction)
        assessment = assessment_builder.assess(reconstruction, lifecycle)
    except (
        DiagnosticAssessmentIntegrityError,
        LifecycleAnalysisIntegrityError,
    ) as exc:
        raise DiagnosticReadIntegrityError(str(exc)) from exc

    return DiagnosticProblemOccurrenceView(
        subject_ref=subject_ref,
        observed_at=occurrence.observed_at,
        strategy_id=occurrence.strategy_id,
        strategy_version=occurrence.strategy_version,
        method=occurrence.method,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=assessment,
        unavailable_reason=None,
    )


def _validate_reconstruction_scope(
    reconstruction: ExecutionReconstruction,
    *,
    subject_ref: object,
) -> None:
    from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef

    if type(subject_ref) is not ProblemGroupingSubjectRef:
        raise TypeError("subject_ref must be ProblemGroupingSubjectRef")

    if reconstruction.tenant_id != subject_ref.tenant_id:
        raise DiagnosticReadIntegrityError(
            "reconstructed tenant_id does not match occurrence subject_ref",
        )
    if reconstruction.task_id != subject_ref.task_id:
        raise DiagnosticReadIntegrityError(
            "reconstructed task_id does not match occurrence subject_ref",
        )
    if reconstruction.run_id != subject_ref.run_id:
        raise DiagnosticReadIntegrityError(
            "reconstructed run_id does not match occurrence subject_ref",
        )


def _is_execution_evidence_unavailable(reconstruction: ExecutionReconstruction) -> bool:
    return (
        reconstruction.is_runtime_history_complete
        and not reconstruction.has_runtime_events
        and not reconstruction.has_transport_evidence
    )


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id


def _validate_bounded_limit(limit: int, *, max_limit: int) -> int:
    if type(limit) is not int or isinstance(limit, bool):
        raise TypeError("limit must be int")
    if limit < 1 or limit > max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")
    return limit
