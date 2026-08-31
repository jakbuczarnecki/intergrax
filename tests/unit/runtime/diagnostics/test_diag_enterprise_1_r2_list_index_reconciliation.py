# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-1-R2 projection reconciliation, telemetry, and cursor-secret proofs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.applications._shared.diagnostic_cursor_secret import (
    _MIN_PROBLEM_LIST_CURSOR_SECRET_BYTES,
    resolve_problem_list_cursor_secret,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemStatus, mint_problem_id
from intergrax.runtime.diagnostics.problem_list_index_reconciliation import (
    ProblemListIndexClassification,
    ProblemListProjectionHealth,
    classify_list_index_projection,
)
from intergrax.runtime.diagnostics.problem_list_query import (
    DecodedListIndexData,
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

_TENANT = "enterprise-1-r2-tenant"
_BASE_TIME = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)
_PARTITION = f"intergrax.diagnostic_problem.v1:{_TENANT}"
_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
_MINIMUM_PROJECTION_AGE = timedelta(hours=1)


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


def _persistence_with_clock(store: InMemoryDocumentStore | None = None):
    persistence = document_store_problem_persistence_for_tests(store)
    persistence.set_clock_for_tests(lambda: _NOW)
    return persistence


def _list_index_document(
    problem,
    *,
    scope,
    projection_written_at: datetime | None = None,
):
    return DocumentRecord(
        partition_key=_PARTITION,
        row_key=list_index_row_key(scope=scope, problem=problem),
        data=encode_list_index_data(
            problem_id=problem.problem_id,
            last_seen_at=problem.last_seen_at,
            status=problem.status,
            record_version=problem.record_version,
            projection_written_at=projection_written_at,
        ),
    )


def _seed_valid_problem(store: InMemoryDocumentStore, *, offset_seconds: int) -> None:
    problem = replace(
        sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id()),
        last_seen_at=_BASE_TIME + timedelta(seconds=offset_seconds),
    )
    store.put(
        DocumentRecord(
            partition_key=_PARTITION,
            row_key=f"record:{problem.problem_id}",
            data=encode_problem_record(problem),
        ),
    )
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )


def _seed_orphan_index(store: InMemoryDocumentStore, *, offset_seconds: int) -> None:
    problem = replace(
        sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id()),
        last_seen_at=_BASE_TIME + timedelta(seconds=offset_seconds),
    )
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )


def test_orphan_before_cutoff_is_not_deleted() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(minutes=5),
            ),
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert page.deleted == 0
    assert page.transient >= 1


def test_orphan_after_cutoff_is_deleted() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert page.deleted == len(list_scopes_for_status(problem.status))
    telemetry = persistence.projection_telemetry_snapshot()
    assert telemetry.deleted_orphan_projection == page.deleted


def test_active_writer_before_cutoff_is_preserved() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(minutes=5),
            ),
        )

    before = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert before.deleted == 0

    store.put(
        DocumentRecord(
            partition_key=_PARTITION,
            row_key=f"record:{problem.problem_id}",
            data=encode_problem_record(problem),
        ),
    )
    page = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert len(page.problems) == 1
    assert page.problems[0].problem_id == problem.problem_id


def test_stale_behind_canonical_is_repaired_after_cutoff() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
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
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert page.repaired >= len(list_scopes_for_status(problem.status))
    listed = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert listed.problems[0].record_version == updated.record_version


def test_index_ahead_of_canonical_repaired_after_cutoff() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)
    ahead = replace(
        problem,
        last_seen_at=_BASE_TIME + timedelta(hours=1),
        record_version=problem.record_version + 1,
    )
    for scope in list_scopes_for_status(ahead.status):
        store.put(
            _list_index_document(
                ahead,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert page.repaired >= len(list_scopes_for_status(ahead.status))
    listed = persistence.query_problems(tenant_id=_TENANT, limit=10)
    assert listed.problems[0].record_version == problem.record_version


def test_same_version_corruption_still_fail_closed() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)
    corrupted = replace(problem, last_seen_at=_BASE_TIME + timedelta(days=3))
    for scope in list_scopes_for_status(corrupted.status):
        store.put(
            _list_index_document(
                corrupted,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )

    with pytest.raises(ProblemPersistenceIntegrityError):
        persistence.query_problems(tenant_id=_TENANT, limit=5)
    telemetry = persistence.projection_telemetry_snapshot()
    assert telemetry.same_version_integrity_failure == 1
    assert persistence.projection_health() is ProblemListProjectionHealth.DEGRADED


def test_read_telemetry_counts_skips() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    for scope in list_scopes_for_status(problem.status):
        store.put(
            _list_index_document(
                problem,
                scope=scope,
                projection_written_at=_NOW - timedelta(hours=2),
            ),
        )

    persistence.query_problems(tenant_id=_TENANT, limit=5)
    telemetry = persistence.projection_telemetry_snapshot()
    assert telemetry.skipped_missing_canonical >= 1


def test_large_garbage_reconciliation_and_read_recovery() -> None:
    store = _CountingDocumentStore()
    persistence = _persistence_with_clock(store)
    for index in range(10_000):
        _seed_orphan_index(store, offset_seconds=100_000 + index)
    for index in range(1_000):
        _seed_valid_problem(store, offset_seconds=index)

    before = persistence.query_problems(tenant_id=_TENANT, limit=100)
    assert len(before.problems) < 100

    deleted_total = 0
    cursor: str | None = None
    while True:
        page = persistence.reconcile_list_indexes(
            tenant_id=_TENANT,
            minimum_projection_age=_MINIMUM_PROJECTION_AGE,
            limit=500,
            cursor=cursor,
        )
        deleted_total += page.deleted
        if not page.has_more:
            break
        cursor = page.next_cursor

    assert deleted_total >= 20_000

    store.query_calls = 0
    store.get_calls = 0
    after = persistence.query_problems(tenant_id=_TENANT, limit=100)
    assert len(after.problems) == 100
    assert store.query_calls <= 2
    assert store.get_calls <= 400


def test_v1_index_decode_and_upgrade_on_reconciliation() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = _persistence_with_clock(store)
    problem = sample_problem(tenant_id=_TENANT, problem_id=mint_problem_id())
    persistence.create(problem)
    for scope in list_scopes_for_status(problem.status):
        store.put(_list_index_document(problem, scope=scope))

    page = persistence.reconcile_list_indexes(
        tenant_id=_TENANT,
        minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        limit=100,
    )
    assert page.consistent + page.repaired >= len(list_scopes_for_status(problem.status))
    assert page.repaired >= len(list_scopes_for_status(problem.status))


def test_classify_transient_without_projection_timestamp() -> None:
    index = DecodedListIndexData(
        problem_id=mint_problem_id(),
        last_seen_at=_BASE_TIME,
        status=ProblemStatus.OPEN,
        record_version=1,
        projection_written_at=None,
        schema_version="intergrax.diagnostic_problem.list_index.v1",
    )
    assert (
        classify_list_index_projection(
            index=index,
            canonical=None,
            now=_NOW,
            minimum_projection_age=_MINIMUM_PROJECTION_AGE,
        )
        is ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN
    )


def test_cursor_secret_missing_fails() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET", raising=False)
    with pytest.raises(ValueError, match="INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"):
        resolve_problem_list_cursor_secret()
    monkeypatch.undo()


def test_cursor_secret_empty_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET", "   ")
    with pytest.raises(ValueError, match="INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"):
        resolve_problem_list_cursor_secret()


def test_cursor_secret_31_bytes_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET", "a" * 31)
    with pytest.raises(ValueError, match="problem_list_cursor_secret_too_short"):
        resolve_problem_list_cursor_secret()


def test_cursor_secret_32_and_64_bytes_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    for length in (32, 64):
        monkeypatch.setenv(
            "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
            "x" * length,
        )
        secret = resolve_problem_list_cursor_secret()
        assert len(secret) == length
        assert len(secret) >= _MIN_PROBLEM_LIST_CURSOR_SECRET_BYTES


def test_cursor_wrong_secret_authentication_failed() -> None:
    codec = ProblemListQueryCursorCodec(secret=TEST_PROBLEM_LIST_CURSOR_SECRET)
    cursor = codec.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(None),
        store_cursor="opaque",
    )
    other = ProblemListQueryCursorCodec(secret=b"other-secret-with-32-byte-minimum-length")
    with pytest.raises(ProblemListQueryCursorError, match="authentication_failed"):
        other.decode(
            cursor,
            tenant_id=_TENANT,
            status_filter=problem_list_scope_for_status(None),
        )


def test_cursor_rotation_invalidates_old_cursor() -> None:
    first = ProblemListQueryCursorCodec(secret=b"first-secret-with-32-byte-minimum-length")
    second = ProblemListQueryCursorCodec(secret=b"second-secret-with-32-byte-minimum-length")
    cursor = first.encode(
        tenant_id=_TENANT,
        status_filter=problem_list_scope_for_status(None),
        store_cursor="opaque",
    )
    with pytest.raises(ProblemListQueryCursorError, match="authentication_failed"):
        second.decode(
            cursor,
            tenant_id=_TENANT,
            status_filter=problem_list_scope_for_status(None),
        )
