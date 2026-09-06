# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_models import (
    DiagnosticScopeDiscoveryIntegrityError,
    DiagnosticScopeDiscoveryRequest,
    DiagnosticScopeDiscoveryStatus,
    ProblemScopeReference,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_provider import (
    DiagnosticScopeProviderResult,
)
from intergrax.runtime.diagnostics.diagnostic_scope_discovery_service import (
    DiagnosticScopeDiscoveryService,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_occurrences, sample_problem
from intergrax.runtime.diagnostics.problem_grouping import (
    problem_grouping_subject_ref_for_application_instance,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemId
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.providers.problem_scope_provider import (
    PROBLEM_SCOPE_PROVIDER_ID,
    ProblemScopeProvider,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    create_problem_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


def _service(
    problem_persistence: ProblemPersistence,
    occurrence_persistence: ProblemOccurrencePersistence,
) -> DiagnosticScopeDiscoveryService:
    return DiagnosticScopeDiscoveryService(
        providers=(
            ProblemScopeProvider(
                problem_persistence=problem_persistence,
                occurrence_persistence=occurrence_persistence,
            ),
        ),
    )


def _request(problem_id: ProblemId) -> DiagnosticScopeDiscoveryRequest:
    return DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=problem_id),
        candidate_limit=10,
    )


def _seed_problem_with_occurrences(
    *,
    subject_refs: tuple[object, ...],
) -> tuple[ProblemPersistence, ProblemOccurrencePersistence, Problem]:
    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem = create_problem_for_tests(
        problem_persistence,
        sample_problem(
            tenant_id=_TENANT,
            subject_refs=subject_refs,
            occurrence_count=0,
        ),
        indexed_subject_refs=subject_refs,
    )
    for occurrence in sample_occurrences(
        subject_refs=subject_refs,
        observed_at=_OBSERVED_AT,
    ):
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
    return problem_persistence, occurrence_persistence, problem


def test_missing_problem_returns_not_found() -> None:
    store = in_memory_document_store_for_problem_tests()
    service = _service(
        document_store_problem_persistence_for_tests(store),
        document_store_occurrence_persistence_for_tests(store),
    )
    result = service.discover_scope(_request(ProblemId("problem_0123456789abcdef0123456789abcdef")))
    assert result.status is DiagnosticScopeDiscoveryStatus.NOT_FOUND


def test_problem_without_occurrences_returns_insufficient_evidence() -> None:
    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = create_problem_for_tests(
        problem_persistence,
        sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=0),
        indexed_subject_refs=(subject,),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE


def test_one_execution_occurrence_returns_resolved() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject,),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.resolved_scope is not None
    assert result.resolved_scope.task_id == subject.task_id
    assert result.resolved_scope.run_id == subject.run_id


def test_repeated_occurrences_same_execution_returns_resolved() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject, subject),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1


def test_two_distinct_execution_scopes_returns_ambiguous() -> None:
    subject_a = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    subject_b = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject_a, subject_b),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.resolved_scope is None
    assert result.candidate_count == 2


def test_only_application_subject_returns_non_execution_subject() -> None:
    subject = problem_grouping_subject_ref_for_application_instance(
        tenant_id=_TENANT,
        application_id="app-a",
        instance_id="instance-a",
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject,),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.NON_EXECUTION_SUBJECT


def test_execution_plus_application_returns_resolved_with_limitation() -> None:
    execution_subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    application_subject = problem_grouping_subject_ref_for_application_instance(
        tenant_id=_TENANT,
        application_id="app-a",
        instance_id="instance-a",
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(execution_subject, application_subject),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert "non-execution" in result.limitations[0]


def test_problem_tenant_mismatch_raises_integrity_error() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=0)
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.return_value = Problem(
        problem_id=problem.problem_id,
        tenant_id="other-tenant",
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=problem.last_seen_at,
        occurrence_count=0,
        record_version=problem.record_version,
        provenance=problem.provenance,
    )
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="tenant_id"):
        _service(problem_persistence, occurrence_persistence).discover_scope(
            _request(problem.problem_id),
        )


def test_occurrence_tenant_mismatch_raises_integrity_error() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=1)
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.return_value = problem
    mismatched_subject = problem_grouping_subject_ref_for_execution(
        tenant_id="other-tenant",
        task_id=subject.task_id,
        run_id=subject.run_id,
    )
    occurrence = sample_occurrences(subject_refs=(mismatched_subject,), observed_at=_OBSERVED_AT)[0]
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    occurrence_persistence.query_occurrences.return_value = ProblemOccurrencePage(
        items=(occurrence,),
        next_cursor=None,
        has_more=False,
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="occurrence"):
        _service(problem_persistence, occurrence_persistence).discover_scope(
            _request(problem.problem_id),
        )


def test_pagination_considers_later_page(monkeypatch: pytest.MonkeyPatch) -> None:
    subject_page_one = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    subject_page_two = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject_page_one, subject_page_two),
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._OCCURRENCE_PAGE_SIZE",
        1,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS


def test_pagination_changes_resolved_to_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    first_subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    second_subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(first_subject, second_subject),
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._OCCURRENCE_PAGE_SIZE",
        1,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2
    assert result.candidate_count_exact is True


def test_two_distinct_execution_scopes_complete_history_is_exact() -> None:
    subject_a = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    subject_b = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject_a, subject_b),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 2
    assert result.candidate_count_exact is True


def test_one_execution_scope_complete_history_is_exact() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject,),
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.RESOLVED
    assert result.candidate_count == 1
    assert result.candidate_count_exact is True


def test_many_execution_scopes_complete_history_is_exact() -> None:
    subjects = tuple(
        problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
        for _ in range(5)
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=subjects,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 5
    assert result.candidate_count_exact is True


def test_candidate_limit_preserves_exact_truth_count() -> None:
    subjects = tuple(
        problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
        for _ in range(5)
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=subjects,
    )
    request = DiagnosticScopeDiscoveryRequest(
        tenant_id=_TENANT,
        reference=ProblemScopeReference(problem_id=problem.problem_id),
        candidate_limit=2,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(request)
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count == 5
    assert result.candidate_count_exact is True
    assert len(result.candidates) == 2


def test_truncated_history_one_scope_is_insufficient_inexact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    extra_subjects = tuple(
        problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
        for _ in range(3)
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject, *extra_subjects),
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._MAX_EXAMINED_OCCURRENCES",
        1,
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._OCCURRENCE_PAGE_SIZE",
        1,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.INSUFFICIENT_EVIDENCE
    assert result.candidate_count_exact is False


def test_truncated_history_two_or_more_scopes_is_ambiguous_inexact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subjects = tuple(
        problem_grouping_subject_ref_for_execution(
            tenant_id=_TENANT,
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
        for _ in range(4)
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=subjects,
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._MAX_EXAMINED_OCCURRENCES",
        2,
    )
    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.providers.problem_scope_provider._OCCURRENCE_PAGE_SIZE",
        1,
    )
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(problem.problem_id),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.AMBIGUOUS
    assert result.candidate_count >= 2
    assert result.candidate_count_exact is False


def test_provider_id_is_frozen() -> None:
    assert PROBLEM_SCOPE_PROVIDER_ID == "problem_scope"


def test_service_unsupported_reference_without_provider() -> None:
    service = DiagnosticScopeDiscoveryService(providers=())
    result = service.discover_scope(_request(ProblemId("problem_0123456789abcdef0123456789abcdef")))
    assert result.status is DiagnosticScopeDiscoveryStatus.UNSUPPORTED_REFERENCE


def test_service_provider_unavailable_on_connection_error() -> None:
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.side_effect = ConnectionError("store down")
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(ProblemId("problem_0123456789abcdef0123456789abcdef")),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_service_provider_unavailable_on_timeout_error() -> None:
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.side_effect = TimeoutError("store timeout")
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    result = _service(problem_persistence, occurrence_persistence).discover_scope(
        _request(ProblemId("problem_0123456789abcdef0123456789abcdef")),
    )
    assert result.status is DiagnosticScopeDiscoveryStatus.PROVIDER_UNAVAILABLE


def test_service_problem_persistence_integrity_maps_to_discovery_integrity() -> None:
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.side_effect = ProblemPersistenceIntegrityError("bad index")
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="bad index"):
        _service(problem_persistence, occurrence_persistence).discover_scope(
            _request(ProblemId("problem_0123456789abcdef0123456789abcdef")),
        )


def test_service_occurrence_persistence_integrity_maps_to_discovery_integrity() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), occurrence_count=1)
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.return_value = problem
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    occurrence_persistence.query_occurrences.side_effect = (
        ProblemOccurrencePersistenceIntegrityError("bad occurrence page")
    )
    with pytest.raises(DiagnosticScopeDiscoveryIntegrityError, match="bad occurrence page"):
        _service(problem_persistence, occurrence_persistence).discover_scope(
            _request(problem.problem_id),
        )


def test_service_unexpected_programming_error_propagates() -> None:
    problem_persistence = MagicMock(spec=ProblemPersistence)
    problem_persistence.get.side_effect = ValueError("programming bug")
    occurrence_persistence = MagicMock(spec=ProblemOccurrencePersistence)
    with pytest.raises(ValueError, match="programming bug"):
        _service(problem_persistence, occurrence_persistence).discover_scope(
            _request(ProblemId("problem_0123456789abcdef0123456789abcdef")),
        )


def test_service_is_deterministic() -> None:
    subject = problem_grouping_subject_ref_for_execution(
        tenant_id=_TENANT,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    problem_persistence, occurrence_persistence, problem = _seed_problem_with_occurrences(
        subject_refs=(subject,),
    )
    service = _service(problem_persistence, occurrence_persistence)
    request = _request(problem.problem_id)
    first = service.discover_scope(request)
    second = service.discover_scope(request)
    assert first == second
