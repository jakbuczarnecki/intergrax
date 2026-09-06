# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ProblemId-backed diagnostic execution scope discovery provider (DG-002 Slice 1)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticExecutionScopeCandidate,
    DiagnosticScopeDiscoveryResult,
    DiagnosticScopeDiscoveryStatus,
    DiagnosticScopeReferenceKind,
    DiagnosticScopeResolutionProvenance,
    ProblemScopeReference,
    build_diagnostic_scope_discovery_result,
    validate_scope_discovery_candidate_limit,
    validate_scope_discovery_tenant_id,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeDiscoveryProvider,
    DiagnosticScopeProviderIntegrityError,
    DiagnosticScopeProviderResult,
    DiagnosticScopeProviderUnavailableError,
    validate_scope_provider_result,
)
from intergrax.runtime.diagnostics.diagnostic_subject import (
    ExecutionDiagnosticSubjectRef,
    validate_execution_diagnostic_subject_ref,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemId, validate_problem_id
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceIntegrityError,
)

PROBLEM_SCOPE_PROVIDER_ID = "problem_scope"

_OCCURRENCE_PAGE_SIZE = 100
_MAX_EXAMINED_OCCURRENCES = 1000
_NON_EXECUTION_LIMITATION = "problem also contains non-execution occurrences"
_TRUNCATION_LIMITATION = (
    "occurrence history exceeded examination bound before scope classification "
    "could be completed"
)


class ProblemScopeProvider:
    """Resolve execution diagnostic scope from tenant-scoped ProblemId."""

    def __init__(
        self,
        *,
        problem_persistence: ProblemPersistence,
        occurrence_persistence: ProblemOccurrencePersistence,
    ) -> None:
        self._problem_persistence = problem_persistence
        self._occurrence_persistence = occurrence_persistence

    @property
    def provider_id(self) -> str:
        return PROBLEM_SCOPE_PROVIDER_ID

    @property
    def supported_reference_kind(self) -> DiagnosticScopeReferenceKind:
        return DiagnosticScopeReferenceKind.PROBLEM

    def discover(
        self,
        *,
        tenant_id: str,
        reference: ProblemScopeReference,
        candidate_limit: int,
    ) -> DiagnosticScopeProviderResult:
        tenant_id = validate_scope_discovery_tenant_id(tenant_id)
        candidate_limit = validate_scope_discovery_candidate_limit(candidate_limit)
        problem_id = validate_problem_id(reference.problem_id)
        provenance = _problem_provenance(problem_id=problem_id)

        problem = _get_problem(
            self._problem_persistence,
            tenant_id=tenant_id,
            problem_id=problem_id,
        )
        if problem is None:
            return _provider_result_from_public(
                build_diagnostic_scope_discovery_result(
                    status=DiagnosticScopeDiscoveryStatus.NOT_FOUND,
                    resolved_scope=None,
                    candidates=(),
                    candidate_count=0,
                    candidate_count_exact=True,
                    provenance=(provenance,),
                ),
            )

        if problem.tenant_id != tenant_id:
            raise DiagnosticScopeProviderIntegrityError(
                "persisted Problem tenant_id does not match discovery request tenant",
            )

        examination = _examine_occurrences(
            self._occurrence_persistence,
            tenant_id=tenant_id,
            problem_id=problem_id,
            provenance=provenance,
        )
        return validate_scope_provider_result(
            _classify_examination(
                examination,
                candidate_limit=candidate_limit,
                provenance=provenance,
            ),
        )


def _get_problem(
    problem_persistence: ProblemPersistence,
    *,
    tenant_id: str,
    problem_id: ProblemId,
) -> Problem | None:
    try:
        return problem_persistence.get(
            tenant_id=tenant_id,
            problem_id=problem_id,
        )
    except ProblemPersistenceIntegrityError as exc:
        raise DiagnosticScopeProviderIntegrityError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DiagnosticScopeProviderUnavailableError(str(exc)) from exc


def _query_occurrences(
    occurrence_persistence: ProblemOccurrencePersistence,
    *,
    tenant_id: str,
    problem_id: ProblemId,
    limit: int,
    cursor: str | None,
) -> ProblemOccurrencePage:
    try:
        return occurrence_persistence.query_occurrences(
            tenant_id=tenant_id,
            problem_id=problem_id,
            limit=limit,
            cursor=cursor,
        )
    except ProblemOccurrencePersistenceIntegrityError as exc:
        raise DiagnosticScopeProviderIntegrityError(str(exc)) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DiagnosticScopeProviderUnavailableError(str(exc)) from exc


def _problem_provenance(
    *,
    problem_id: ProblemId,
) -> DiagnosticScopeResolutionProvenance:
    return DiagnosticScopeResolutionProvenance(
        provider_id=PROBLEM_SCOPE_PROVIDER_ID,
        reference_kind=DiagnosticScopeReferenceKind.PROBLEM,
        canonical_record_ref=f"problem:{problem_id}",
    )


def _provider_result_from_public(
    result: DiagnosticScopeDiscoveryResult,
) -> DiagnosticScopeProviderResult:
    return DiagnosticScopeProviderResult(
        status=result.status,
        resolved_scope=result.resolved_scope,
        candidates=result.candidates,
        candidate_count=result.candidate_count,
        candidate_count_exact=result.candidate_count_exact,
        provenance=result.provenance,
        limitations=result.limitations,
    )


class _OccurrenceExamination:
    def __init__(self) -> None:
        self.execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate] = {}
        self.has_non_execution = False
        self.total_occurrences = 0
        self.examination_truncated = False
        self.candidate_count_exact = True


def _examine_occurrences(
    occurrence_persistence: ProblemOccurrencePersistence,
    *,
    tenant_id: str,
    problem_id: ProblemId,
    provenance: DiagnosticScopeResolutionProvenance,
) -> _OccurrenceExamination:
    examination = _OccurrenceExamination()
    cursor: str | None = None
    page_size = min(_OCCURRENCE_PAGE_SIZE, _MAX_EXAMINED_OCCURRENCES)
    page: ProblemOccurrencePage | None = None

    while examination.total_occurrences < _MAX_EXAMINED_OCCURRENCES:
        remaining = _MAX_EXAMINED_OCCURRENCES - examination.total_occurrences
        limit = min(page_size, remaining)
        page = _query_occurrences(
            occurrence_persistence,
            tenant_id=tenant_id,
            problem_id=problem_id,
            limit=limit,
            cursor=cursor,
        )
        for occurrence in page.items:
            examination.total_occurrences += 1
            subject_ref = occurrence.subject_ref
            if subject_ref.tenant_id != tenant_id:
                raise DiagnosticScopeProviderIntegrityError(
                    "occurrence subject_ref tenant_id does not match discovery request tenant",
                )
            execution_ref = subject_ref.execution()
            if execution_ref is None:
                examination.has_non_execution = True
                continue
            validated_execution = validate_execution_diagnostic_subject_ref(execution_ref)
            identity = (str(validated_execution.task_id), str(validated_execution.run_id))
            if identity not in examination.execution_scopes:
                examination.execution_scopes[identity] = DiagnosticExecutionScopeCandidate(
                    subject_ref=validated_execution,
                    provenance=provenance,
                )

        if not page.has_more:
            break

        if examination.total_occurrences >= _MAX_EXAMINED_OCCURRENCES:
            examination.examination_truncated = True
            examination.candidate_count_exact = False
            break

        cursor = page.next_cursor

    if page is not None and page.has_more:
        examination.examination_truncated = True
        examination.candidate_count_exact = False

    return examination


def _classify_examination(
    examination: _OccurrenceExamination,
    *,
    candidate_limit: int,
    provenance: DiagnosticScopeResolutionProvenance,
) -> DiagnosticScopeProviderResult:
    limitations: list[str] = []
    ordered_candidates = _ordered_execution_candidates(examination.execution_scopes)
    distinct_count = len(ordered_candidates)

    if examination.total_occurrences == 0:
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE,
                resolved_scope=None,
                candidates=(),
                candidate_count=0,
                candidate_count_exact=True,
                provenance=(provenance,),
                limitations=("problem exists but has no occurrences",),
            ),
        )

    if examination.examination_truncated and distinct_count <= 1:
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE,
                resolved_scope=None,
                candidates=tuple(ordered_candidates[:candidate_limit]),
                candidate_count=distinct_count,
                candidate_count_exact=False,
                provenance=(provenance,),
                limitations=(_TRUNCATION_LIMITATION,),
            ),
        )

    if distinct_count == 0:
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.NON_EXECUTION_SUBJECT,
                resolved_scope=None,
                candidates=(),
                candidate_count=0,
                candidate_count_exact=True,
                provenance=(provenance,),
            ),
        )

    if distinct_count == 1:
        if examination.has_non_execution:
            limitations.append(_NON_EXECUTION_LIMITATION)
        resolved_scope = ordered_candidates[0].subject_ref
        return _provider_result_from_public(
            build_diagnostic_scope_discovery_result(
                status=DiagnosticScopeDiscoveryStatus.RESOLVED,
                resolved_scope=resolved_scope,
                candidates=(ordered_candidates[0],),
                candidate_count=1,
                candidate_count_exact=examination.candidate_count_exact,
                provenance=(provenance,),
                limitations=tuple(limitations),
            ),
        )

    return _provider_result_from_public(
        build_diagnostic_scope_discovery_result(
            status=DiagnosticScopeDiscoveryStatus.AMBIGUOUS,
            resolved_scope=None,
            candidates=tuple(ordered_candidates[:candidate_limit]),
            candidate_count=distinct_count,
            candidate_count_exact=examination.candidate_count_exact,
            provenance=(provenance,),
            limitations=tuple(limitations),
        ),
    )


def _ordered_execution_candidates(
    execution_scopes: dict[tuple[str, str], DiagnosticExecutionScopeCandidate],
) -> list[DiagnosticExecutionScopeCandidate]:
    return sorted(
        execution_scopes.values(),
        key=lambda candidate: (
            str(candidate.subject_ref.task_id),
            str(candidate.subject_ref.run_id),
        ),
    )
