# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1 scalable bounded Problem list query proofs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemStatus,
    mint_problem_id,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    ProblemListQueryCursorCodec,
    ProblemListQueryCursorError,
    encode_list_index_data,
    list_index_row_key,
    list_scopes_for_status,
    problem_list_scope_for_status,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceIntegrityError
from intergrax.runtime.diagnostics.problem_record_codec import encode_problem_record
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_DOCUMENT_STORE_CURSOR_SECRET,
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "enterprise-1-tenant"
_OTHER_TENANT = "enterprise-1-other"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


class _CountingDocumentStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__(cursor_secret=TEST_DOCUMENT_STORE_CURSOR_SECRET)
        self.query_calls = 0
        self.get_calls = 0

    def query(self, partition_key: str, *, limit: int, row_key_prefix: str | None = None, cursor: str | None = None):
        self.query_calls += 1
        return super().query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def get(self, partition_key: str, row_key: str):
        self.get_calls += 1
        return super().get(partition_key, row_key)


def _seed_many_indexed_problems(store: InMemoryDocumentStore, count: int) -> None:
    partition_key = f"intergrax.diagnostic_problem.v1:{_TENANT}"
    for index in range(count):
        problem = replace(
            sample_problem(
                tenant_id=_TENANT,
                problem_id=mint_problem_id(),
                subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            ),
            last_seen_at=_BASE_TIME + timedelta(seconds=index),
            status=ProblemStatus.OPEN if index % 2 == 0 else ProblemStatus.RESOLVED,
        )
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=f"record:{problem.problem_id}",
                data=encode_problem_record(problem),
            ),
        )
        for scope in list_scopes_for_status(problem.status):
            store.put(
                DocumentRecord(
                    partition_key=partition_key,
                    row_key=list_index_row_key(scope=scope, problem=problem),
                    data=encode_list_index_data(
                        problem_id=problem.problem_id,
                        last_seen_at=problem.last_seen_at,
                        status=problem.status,
                        record_version=problem.record_version,
                    ),
                ),
            )


def _seed_pagination_fixture(store: InMemoryDocumentStore) -> None:
    partition_key = f"intergrax.diagnostic_problem.v1:{_TENANT}"
    timestamps = (_BASE_TIME, _BASE_TIME, _BASE_TIME + timedelta(hours=1))
    for observed_at in timestamps:
        problem = replace(
            sample_problem(
                tenant_id=_TENANT,
                problem_id=mint_problem_id(),
                subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            ),
            last_seen_at=observed_at,
            status=ProblemStatus.OPEN,
        )
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=f"record:{problem.problem_id}",
                data=encode_problem_record(problem),
            ),
        )
        for scope in list_scopes_for_status(problem.status):
            store.put(
                DocumentRecord(
                    partition_key=partition_key,
                    row_key=list_index_row_key(scope=scope, problem=problem),
                    data=encode_list_index_data(
                        problem_id=problem.problem_id,
                        last_seen_at=problem.last_seen_at,
                        status=problem.status,
                        record_version=problem.record_version,
                    ),
                ),
            )


def test_bounded_query_does_not_materialize_full_tenant() -> None:
    store = _CountingDocumentStore()
    _seed_many_indexed_problems(store, 10_000)
    persistence = document_store_problem_persistence_for_tests(store)

    page = persistence.query_problems(tenant_id=_TENANT, limit=100)

    assert len(page.problems) == 100
    assert page.has_more is True
    assert page.next_cursor is not None
    assert store.query_calls <= 2
    assert store.get_calls <= 400


def test_pagination_no_duplicates_and_deterministic_order() -> None:
    store = in_memory_document_store_for_problem_tests()
    _seed_pagination_fixture(store)
    persistence = document_store_problem_persistence_for_tests(store)

    page1 = persistence.query_problems(tenant_id=_TENANT, limit=2)
    page2 = persistence.query_problems(
        tenant_id=_TENANT,
        limit=2,
        cursor=page1.next_cursor,
    )

    combined = [*page1.problems, *page2.problems]
    assert len(combined) == 3
    assert len({problem.problem_id for problem in combined}) == 3
    assert combined[0].last_seen_at >= combined[1].last_seen_at >= combined[2].last_seen_at
    tied = [problem for problem in combined if problem.last_seen_at == _BASE_TIME]
    assert [str(problem.problem_id) for problem in tied] == sorted(
        str(problem.problem_id) for problem in tied
    )


def test_status_filter_across_pages() -> None:
    store = in_memory_document_store_for_problem_tests()
    _seed_many_indexed_problems(store, 6)
    persistence = document_store_problem_persistence_for_tests(store)

    collected: list[Problem] = []
    cursor: str | None = None
    while True:
        page = persistence.query_problems(
            tenant_id=_TENANT,
            status=ProblemStatus.OPEN,
            limit=2,
            cursor=cursor,
        )
        collected.extend(page.problems)
        if not page.has_more:
            break
        cursor = page.next_cursor

    assert len(collected) == 3
    assert all(problem.status is ProblemStatus.OPEN for problem in collected)


def test_cursor_tenant_binding_rejects_cross_tenant() -> None:
    persistence = InMemoryProblemPersistence()
    persistence.create(
        sample_problem(
            tenant_id=_TENANT,
            problem_id=mint_problem_id(),
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        ),
    )
    page = persistence.query_problems(tenant_id=_TENANT, limit=1)
    assert page.next_cursor is None

    codec = ProblemListQueryCursorCodec(secret=TEST_PROBLEM_LIST_CURSOR_SECRET)
    forged = codec.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(None),
        store_cursor="0",
    )
    with pytest.raises(ProblemListQueryCursorError, match="tenant_mismatch"):
        codec.decode(
            forged,
            tenant_id=_OTHER_TENANT,
            status_filter=problem_list_scope_for_status(None),
        )


def test_cursor_status_binding_rejects_mismatched_filter() -> None:
    persistence = InMemoryProblemPersistence()
    persistence.create(
        sample_problem(
            tenant_id=_TENANT,
            problem_id=mint_problem_id(),
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        ),
    )
    page = persistence.query_problems(
        tenant_id=_TENANT,
        status=ProblemStatus.OPEN,
        limit=1,
    )
    assert page.next_cursor is None

    codec = ProblemListQueryCursorCodec(secret=TEST_PROBLEM_LIST_CURSOR_SECRET)
    forged = codec.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(ProblemStatus.OPEN),
        store_cursor="0",
    )
    with pytest.raises(ProblemListQueryCursorError, match="status_mismatch"):
        codec.decode(
            forged,
            tenant_id=_TENANT,
            status_filter=problem_list_scope_for_status(ProblemStatus.RESOLVED),
        )


def test_tampered_cursor_is_rejected() -> None:
    codec = ProblemListQueryCursorCodec(secret=b"enterprise-1-secret")
    cursor = codec.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(None),
        store_cursor="1",
    )
    tampered = cursor[:-1] + ("a" if cursor[-1] != "a" else "b")
    with pytest.raises(ProblemListQueryCursorError):
        codec.decode(
            tampered,
            tenant_id=_TENANT,
            status_filter=problem_list_scope_for_status(None),
        )


def test_list_index_missing_canonical_is_skipped_without_integrity_error() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(
        tenant_id=_TENANT,
        problem_id=mint_problem_id(),
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
    )
    persistence.create(problem)

    partition_key = f"intergrax.diagnostic_problem.v1:{_TENANT}"
    store.delete(partition_key, f"record:{problem.problem_id}")

    page = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert page.problems == ()
    assert page.has_more is False


def test_stale_list_index_metadata_at_same_version_raises_integrity_error() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(
        tenant_id=_TENANT,
        problem_id=mint_problem_id(),
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
    )
    persistence.create(problem)

    partition_key = f"intergrax.diagnostic_problem.v1:{_TENANT}"
    for scope in list_scopes_for_status(problem.status):
        row_key = list_index_row_key(scope=scope, problem=problem)
        stale_data = encode_list_index_data(
            problem_id=problem.problem_id,
            last_seen_at=_BASE_TIME + timedelta(days=1),
            status=problem.status,
            record_version=problem.record_version,
        )
        existing = store.get(partition_key, row_key)
        assert existing is not None
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=row_key,
                data=stale_data,
            ),
        )

    with pytest.raises(
        ProblemPersistenceIntegrityError,
        match="metadata inconsistent with canonical Problem",
    ):
        persistence.query_problems(tenant_id=_TENANT, limit=10)


def test_update_changes_list_index_without_full_rescan() -> None:
    store = _CountingDocumentStore()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(
        tenant_id=_TENANT,
        problem_id=mint_problem_id(),
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
    )
    persistence.create(problem)
    store.query_calls = 0
    store.get_calls = 0

    updated = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(hours=2),
        status=ProblemStatus.RESOLVED,
        record_version=problem.record_version + 1,
    )
    persistence.update(updated, expected_version=problem.record_version)
    store.query_calls = 0
    store.get_calls = 0

    page = persistence.query_problems(tenant_id=_TENANT, status=ProblemStatus.RESOLVED, limit=10)
    assert len(page.problems) == 1
    assert page.problems[0].problem_id == problem.problem_id
    assert store.query_calls == 1
    assert store.get_calls <= len(page.problems)
