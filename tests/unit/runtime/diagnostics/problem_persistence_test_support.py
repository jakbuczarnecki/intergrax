# © Artur Czarnecki. All rights reserved.

"""Shared test helpers for diagnostic Problem persistence suites."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    DocumentStoreProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemLifecycleEngine, ProblemId
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.persistence_conformance import sample_subject_refs
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef

TEST_PROBLEM_LIST_CURSOR_SECRET = b"deterministic-test-problem-list-cursor-v1"
TEST_DOCUMENT_STORE_CURSOR_SECRET = b"deterministic-test-document-store-cursor-v1"


def in_memory_document_store_for_problem_tests() -> InMemoryDocumentStore:
    return InMemoryDocumentStore(cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET)


def document_store_problem_persistence_for_tests(
    document_store: ConditionalDocumentStore | None = None,
) -> DocumentStoreProblemPersistence:
    store = document_store or in_memory_document_store_for_problem_tests()
    return DocumentStoreProblemPersistence(
        store,
        list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
        document_query_cursor_codec=store.query_cursor_codec,
    )


def document_store_occurrence_persistence_for_tests(
    document_store: ConditionalDocumentStore | None = None,
) -> DocumentStoreProblemOccurrencePersistence:
    store = document_store or in_memory_document_store_for_problem_tests()
    return DocumentStoreProblemOccurrencePersistence(
        store,
        occurrence_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
        document_query_cursor_codec=store.query_cursor_codec,
    )


def in_memory_lifecycle_engine_for_tests() -> ProblemLifecycleEngine:
    """Lifecycle engine with in-memory Problem store and DocumentStore occurrence history."""
    occurrence_store = in_memory_document_store_for_problem_tests()
    return ProblemLifecycleEngine(
        InMemoryProblemPersistence(),
        document_store_occurrence_persistence_for_tests(occurrence_store),
    )


def document_store_lifecycle_stack_for_tests() -> tuple[
    InMemoryDocumentStore,
    DocumentStoreProblemPersistence,
    DocumentStoreProblemOccurrencePersistence,
    ProblemLifecycleEngine,
]:
    store = in_memory_document_store_for_problem_tests()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    engine = lifecycle_engine_for_tests(
        problem_persistence,
        occurrence_persistence,
        document_store=store,
    )
    return store, problem_persistence, occurrence_persistence, engine


def query_all_occurrences_for_problem(
    occurrence_persistence: ProblemOccurrencePersistence,
    *,
    tenant_id: str,
    problem_id: object,
    page_limit: int = 500,
):
    from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId

    occurrences = []
    cursor = None
    while True:
        page = occurrence_persistence.query_occurrences(
            tenant_id=tenant_id,
            problem_id=ProblemId(str(problem_id)),
            limit=page_limit,
            cursor=cursor,
        )
        occurrences.extend(page.items)
        if not page.has_more:
            break
        cursor = page.next_cursor
    return tuple(occurrences)


def lifecycle_engine_for_tests(
    problem_persistence: ProblemPersistence,
    occurrence_persistence: ProblemOccurrencePersistence | None = None,
    *,
    document_store: ConditionalDocumentStore | None = None,
) -> ProblemLifecycleEngine:
    store = document_store or in_memory_document_store_for_problem_tests()
    resolved_occurrence = (
        occurrence_persistence
        or document_store_occurrence_persistence_for_tests(store)
    )
    return ProblemLifecycleEngine(
        problem_persistence,
        resolved_occurrence,
    )


def read_service_for_tests(
    problem_persistence: ProblemPersistence,
    execution_reconstructor: object,
    *,
    occurrence_persistence: ProblemOccurrencePersistence | None = None,
    document_store: ConditionalDocumentStore | None = None,
):
    from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService

    store = document_store or in_memory_document_store_for_problem_tests()
    return DiagnosticReadService(
        problem_persistence=problem_persistence,
        occurrence_persistence=(
            occurrence_persistence
            or document_store_occurrence_persistence_for_tests(store)
        ),
        execution_reconstructor=execution_reconstructor,
    )


def query_all_problems_for_tenant(
    persistence: ProblemPersistence,
    tenant_id: str,
    *,
    page_limit: int = 500,
) -> tuple[Problem, ...]:
    """Conformance/test helper — paginated tenant materialization (not production API)."""
    problems: list[Problem] = []
    cursor: str | None = None
    while True:
        page = persistence.query_problems(
            tenant_id=tenant_id,
            limit=page_limit,
            cursor=cursor,
        )
        problems.extend(page.problems)
        if not page.has_more:
            break
        cursor = page.next_cursor
    problems.sort(key=lambda item: str(item.problem_id))
    return tuple(problems)


def create_problem_for_tests(
    persistence: ProblemPersistence,
    record: Problem,
    *,
    indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] | None = None,
) -> Problem:
    """Create with durable subject ownership index seeds (defaults to one sample ref)."""
    refs = (
        indexed_subject_refs
        if indexed_subject_refs is not None
        else sample_subject_refs(record)
    )
    return persistence.create(record, indexed_subject_refs=refs)


def update_problem_for_tests(
    persistence: ProblemPersistence,
    record: Problem,
    *,
    expected_version: int,
    indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
) -> Problem:
    return persistence.update(
        record,
        expected_version=expected_version,
        indexed_subject_refs=indexed_subject_refs,
    )
