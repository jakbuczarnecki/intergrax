# © Artur Czarnecki. All rights reserved.

"""Shared test helpers for diagnostic Problem persistence suites."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence

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
