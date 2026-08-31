# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R1 deterministic read-index transition proofs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier, Thread

import pytest

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus, mint_problem_id
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
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "enterprise-1-r1-tenant"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_PARTITION = f"intergrax.diagnostic_problem.v1:{_TENANT}"


def _list_index_document(problem, *, scope):
    return DocumentRecord(
        partition_key=_PARTITION,
        row_key=list_index_row_key(scope=scope, problem=problem),
        data=encode_list_index_data(
            problem_id=problem.problem_id,
            last_seen_at=problem.last_seen_at,
            status=problem.status,
            record_version=problem.record_version,
        ),
    )


def test_race_a_create_index_visible_before_canonical_is_skipped() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(
        tenant_id=_TENANT,
        problem_id=mint_problem_id(),
    )
    for scope in list_scopes_for_status(problem.status):
        store.put(_list_index_document(problem, scope=scope))

    page = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert page.problems == ()
    assert page.has_more is False

    store.put(
        DocumentRecord(
            partition_key=_PARTITION,
            row_key=f"record:{problem.problem_id}",
            data=encode_problem_record(problem),
        ),
    )
    page_after = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert len(page_after.problems) == 1
    assert page_after.problems[0].problem_id == problem.problem_id


def test_race_b_new_index_before_canonical_update_is_skipped() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)

    leading = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(hours=1),
        record_version=problem.record_version + 1,
    )
    for scope in list_scopes_for_status(leading.status):
        store.put(_list_index_document(leading, scope=scope))

    page = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert len(page.problems) == 1
    assert page.problems[0].record_version == problem.record_version
    assert page.problems[0].last_seen_at == problem.last_seen_at


def test_race_c_stale_index_after_canonical_update_is_skipped() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)

    updated = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(hours=2),
        record_version=problem.record_version + 1,
    )
    store.put(
        DocumentRecord(
            partition_key=_PARTITION,
            row_key=f"record:{updated.problem_id}",
            data=encode_problem_record(updated),
        ),
    )
    for scope in list_scopes_for_status(updated.status):
        store.put(_list_index_document(updated, scope=scope))

    page = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert len(page.problems) == 1
    assert page.problems[0].record_version == updated.record_version


def test_race_d_writer_death_leaves_skippable_projection_then_repair_on_write() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(_list_index_document(problem, scope=scope))

    assert persistence.query_problems(tenant_id=_TENANT, limit=5).problems == ()

    repaired = persistence.create(problem)
    page = persistence.query_problems(tenant_id=_TENANT, limit=5)
    assert len(page.problems) == 1
    assert page.problems[0] == repaired


def test_corruption_same_version_metadata_mismatch_still_fail_closed() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)

    corrupted = replace(problem, last_seen_at=_BASE_TIME + timedelta(days=3))
    for scope in list_scopes_for_status(corrupted.status):
        store.put(_list_index_document(corrupted, scope=scope))

    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.query_problems(tenant_id=_TENANT, limit=5)


def test_cursor_missing_production_secret_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET", raising=False)
    with pytest.raises(ValueError, match="INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"):
        from intergrax.applications._shared.diagnostic_cursor_secret import (
            resolve_problem_list_cursor_secret,
        )

        resolve_problem_list_cursor_secret()


def test_cursor_wrong_secret_rejected() -> None:
    codec = ProblemListQueryCursorCodec(secret=b"expected-secret")
    cursor = codec.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(None),
        store_cursor="opaque",
    )
    other = ProblemListQueryCursorCodec(secret=b"other-secret")
    with pytest.raises(ProblemListQueryCursorError, match="authentication_failed"):
        other.decode(
            cursor,
            tenant_id=_TENANT,
            status_filter=problem_list_scope_for_status(None),
        )


def test_concurrent_reader_during_update_no_false_integrity_error() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)
    updated = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(minutes=5),
        status=ProblemStatus.RESOLVED,
        record_version=problem.record_version + 1,
    )

    errors: list[BaseException] = []
    start = Barrier(2)

    def reader() -> None:
        start.wait()
        for _ in range(50):
            try:
                persistence.query_problems(tenant_id=_TENANT, limit=5)
            except ProblemPersistenceIntegrityError as exc:
                errors.append(exc)

    def writer() -> None:
        start.wait()
        persistence.update(updated, expected_version=problem.record_version)

    threads = [Thread(target=reader), Thread(target=writer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
