# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R3 exactly-once occurrence stats contribution proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    DocumentStoreProblemOccurrencePersistence,
    _STATS_ROW_KEY,
    _occurrence_partition,
    _occurrence_row_key,
    _stats_contribution_row_key,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
)
from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
    encode_occurrence_stats_contribution_marker,
    encode_occurrence_stats_record,
    encode_problem_occurrence_record,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_lifecycle_stack_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-r3"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)


class _FailStatsMergeOnceStore:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._stats_failures_remaining = 1

    @property
    def query_cursor_codec(self):
        return self._delegate.query_cursor_codec

    def get(self, partition_key: str, row_key: str):
        return self._delegate.get(partition_key, row_key)

    def put(self, document) -> None:
        self._delegate.put(document)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._delegate.delete(partition_key, row_key)

    def query(self, partition_key: str, *, limit: int = 100, row_key_prefix=None, cursor=None):
        return self._delegate.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def close(self) -> None:
        self._delegate.close()

    def put_if_absent(self, document) -> bool:
        if self._stats_failures_remaining > 0 and document.row_key == _STATS_ROW_KEY:
            self._stats_failures_remaining -= 1
            raise RuntimeError("simulated stats merge failure")
        return self._delegate.put_if_absent(document)

    def replace_if_match(self, *, expected, replacement) -> bool:
        if self._stats_failures_remaining > 0 and replacement.row_key == _STATS_ROW_KEY:
            self._stats_failures_remaining -= 1
            raise RuntimeError("simulated stats merge failure")
        return self._delegate.replace_if_match(
            expected=expected,
            replacement=replacement,
        )

    def delete_if_match(self, *, expected) -> bool:
        return self._delegate.delete_if_match(expected=expected)


from intergrax.integrations.contracts.document_store import ConditionalDocumentStore

ConditionalDocumentStore.register(_FailStatsMergeOnceStore)


def _seed_n_occurrences(
    persistence,
    *,
    problem,
    count: int,
) -> None:
    for _ in range(count):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT,
        )[0]
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )


def test_r2_blocker_stats_99_marker_exists_increment_failed_retry_converges() -> None:
    """Exact R2 blocker: stats=99, marker exists, increment failed → retry reaches 100."""
    from intergrax.integrations.contracts.document_store import DocumentRecord

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    _seed_n_occurrences(persistence, problem=problem, count=99)

    occurrence = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=100),
    )[0]
    occurrence_id = problem_occurrence_id_for(occurrence)
    partition = _occurrence_partition(_TENANT, problem.problem_id)

    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_occurrence_row_key(occurrence, occurrence_id=occurrence_id),
            data=encode_problem_occurrence_record(occurrence),
        ),
    )
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_stats_contribution_row_key(occurrence_id),
            data=encode_occurrence_stats_contribution_marker(
                occurrence_id=occurrence_id,
                observed_at=occurrence.observed_at,
                stats_count_snapshot=99,
            ),
        ),
    )

    stats_before = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats_before is not None
    assert stats_before.occurrence_count == 99

    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    stats_after = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats_after is not None
    assert stats_after.occurrence_count == 100


def test_x2_increment_succeeds_before_marker_finalize_no_double_count() -> None:
    """X2: stats incremented, marker still pending → retry must not double count."""
    from intergrax.integrations.contracts.document_store import DocumentRecord

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    occurrence_id = problem_occurrence_id_for(occurrence)
    partition = _occurrence_partition(_TENANT, problem.problem_id)

    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_occurrence_row_key(occurrence, occurrence_id=occurrence_id),
            data=encode_problem_occurrence_record(occurrence),
        ),
    )
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_stats_contribution_row_key(occurrence_id),
            data=encode_occurrence_stats_contribution_marker(
                occurrence_id=occurrence_id,
                observed_at=occurrence.observed_at,
                stats_count_snapshot=0,
            ),
        ),
    )
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_STATS_ROW_KEY,
            data=encode_occurrence_stats_record(
                occurrence_count=1,
                first_seen_at=occurrence.observed_at,
                last_seen_at=occurrence.observed_at,
                last_committed_occurrence_id=str(occurrence_id),
            ),
        ),
    )

    for _ in range(5):
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )
    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


def test_concurrent_duplicate_append_100_writers_exactly_once() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    barrier = threading.Barrier(100)
    results: list[ProblemOccurrenceAppendResult] = []

    def _append() -> None:
        barrier.wait(timeout=30)
        results.append(
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            ),
        )

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_append) for _ in range(100)]
        for future in futures:
            future.result(timeout=60)

    assert results.count(ProblemOccurrenceAppendResult.CREATED) == 1
    assert results.count(ProblemOccurrenceAppendResult.ALREADY_EXISTS) == 99
    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


def test_concurrent_distinct_append_100_writers_exact_count() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    occurrences = tuple(
        sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        for index in range(100)
    )
    barrier = threading.Barrier(100)

    def _append(occurrence) -> None:
        barrier.wait(timeout=30)
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_append, occurrence) for occurrence in occurrences]
        for future in futures:
            future.result(timeout=120)

    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 100
    assert stats.first_seen_at == min(occurrence.observed_at for occurrence in occurrences)
    assert stats.last_seen_at == max(occurrence.observed_at for occurrence in occurrences)


def test_out_of_order_timestamps_preserve_first_last_and_count() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    same_time = _OBSERVED_AT + timedelta(hours=1)
    older = same_time - timedelta(minutes=5)
    newer = same_time + timedelta(minutes=5)
    occurrences = (
        sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=newer,
        )[0],
        sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=older,
        )[0],
        sample_occurrences(
            subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
            observed_at=same_time,
        )[0],
    )
    barrier = threading.Barrier(3)

    def _append(occurrence) -> None:
        barrier.wait(timeout=10)
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_append, occurrence) for occurrence in occurrences]
        for future in futures:
            future.result(timeout=30)

    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 3
    assert stats.first_seen_at == older
    assert stats.last_seen_at == newer


class _CrashAfterWriteStore:
    """Fail after selected write boundaries to simulate process crash."""

    def __init__(self, delegate, *, fail_after_row_key: str) -> None:
        self._delegate = delegate
        self._fail_after_row_key = fail_after_row_key
        self._failed = False

    @property
    def query_cursor_codec(self):
        return self._delegate.query_cursor_codec

    def get(self, partition_key: str, row_key: str):
        return self._delegate.get(partition_key, row_key)

    def put(self, document) -> None:
        self._delegate.put(document)

    def delete(self, partition_key: str, row_key: str) -> None:
        self._delegate.delete(partition_key, row_key)

    def query(self, partition_key: str, *, limit: int = 100, row_key_prefix=None, cursor=None):
        return self._delegate.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
        )

    def close(self) -> None:
        self._delegate.close()

    def _maybe_fail(self, row_key: str) -> None:
        if not self._failed and row_key == self._fail_after_row_key:
            self._failed = True
            raise RuntimeError(f"simulated crash after write to {row_key}")

    def put_if_absent(self, document) -> bool:
        created = self._delegate.put_if_absent(document)
        if created:
            self._maybe_fail(document.row_key)
        return created

    def replace_if_match(self, *, expected, replacement) -> bool:
        replaced = self._delegate.replace_if_match(
            expected=expected,
            replacement=replacement,
        )
        if replaced:
            self._maybe_fail(replacement.row_key)
        return replaced

    def delete_if_match(self, *, expected) -> bool:
        return self._delegate.delete_if_match(expected=expected)


ConditionalDocumentStore.register(_CrashAfterWriteStore)


@pytest.mark.parametrize(
    "fail_after_row_key",
    [
        "occ:",
        "meta:stats_contrib:",
        _STATS_ROW_KEY,
    ],
)
def test_crash_prefix_matrix_converges_after_retry(fail_after_row_key: str) -> None:
    base = in_memory_document_store_for_problem_tests()
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    occurrence_id = problem_occurrence_id_for(occurrence)
    if fail_after_row_key == "occ:":
        boundary = _occurrence_row_key(occurrence, occurrence_id=occurrence_id)
    elif fail_after_row_key == "meta:stats_contrib:":
        boundary = _stats_contribution_row_key(occurrence_id)
    else:
        boundary = fail_after_row_key

    store = _CrashAfterWriteStore(base, fail_after_row_key=boundary)
    persistence = document_store_occurrence_persistence_for_tests(store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    restarted = document_store_occurrence_persistence_for_tests(base)
    assert (
        restarted.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        in {
            ProblemOccurrenceAppendResult.CREATED,
            ProblemOccurrenceAppendResult.ALREADY_EXISTS,
        }
    )
    stats = restarted.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


def test_100k_with_injected_failures_matches_row_and_stats_counts() -> None:
    base = in_memory_document_store_for_problem_tests()
    store = _FailStatsMergeOnceStore(base)
    persistence = document_store_occurrence_persistence_for_tests(store)
    _, problem_persistence, _, _ = document_store_lifecycle_stack_for_tests()
    del problem_persistence
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0)

    for index in range(100_000):
        if index % 997 == 0 and index > 0:
            store._stats_failures_remaining = 1
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        for attempt in range(3):
            try:
                persistence.append_if_absent(
                    tenant_id=_TENANT,
                    problem_id=problem.problem_id,
                    occurrence=occurrence,
                )
                break
            except RuntimeError:
                if attempt == 2:
                    raise

    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 100_000

    page = persistence.query_occurrences(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        limit=100,
    )
    assert len(page.items) == 100
    assert page.has_more is True


def test_migration_stats_failure_at_501_restart_converges_to_1000() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_persistence import (
        _document_partition,
        _record_row_key,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_migration import (
        migrate_legacy_problem_inline_occurrences,
        verify_legacy_occurrences_migrated,
    )
    from intergrax.runtime.diagnostics.problem_record_codec import (
        _encode_legacy_problem_payload_v1,
    )

    base = in_memory_document_store_for_problem_tests()
    store = _FailStatsMergeOnceStore(base)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    partition = _document_partition(_TENANT)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=1000)
    subject_refs = tuple(_sample_subject_ref(tenant_id=_TENANT) for _ in range(1000))
    occurrences = sample_occurrences(subject_refs=subject_refs, observed_at=_OBSERVED_AT)
    bounded = sample_problem(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        subject_refs=(subject_refs[0],),
        occurrence_count=1000,
        observed_at=_OBSERVED_AT,
    )
    store.put_if_absent(
        DocumentRecord(
            partition_key=partition,
            row_key=_record_row_key(problem.problem_id),
            data={
                "schema_version": "intergrax.diagnostic_problem.persistence.v1",
                "payload": _encode_legacy_problem_payload_v1(
                    problem=bounded,
                    current_subject_refs=subject_refs,
                    occurrences=occurrences,
                ),
            },
        ),
    )

    store._stats_failures_remaining = 1
    with pytest.raises(RuntimeError, match="simulated stats merge failure"):
        migrate_legacy_problem_inline_occurrences(
            tenant_id=_TENANT,
            problem_persistence=problem_persistence,
            occurrence_persistence=occurrence_persistence,
            document_store=store,
            limit=1000,
        )

    restarted = document_store_occurrence_persistence_for_tests(base)
    page = migrate_legacy_problem_inline_occurrences(
        tenant_id=_TENANT,
        problem_persistence=problem_persistence,
        occurrence_persistence=restarted,
        document_store=base,
        limit=1000,
    )
    assert len(page.migrated_problem_ids) == 1
    stats = restarted.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1000
    assert verify_legacy_occurrences_migrated(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence_persistence=restarted,
        document_store=base,
    )
