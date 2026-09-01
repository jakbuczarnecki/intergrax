# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R6 storage capability feasibility and atomic append proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemOccurrenceAggregateHealth
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    mark_problem_reconciliation_required,
    reconcile_problem_occurrence_aggregate,
    scan_occurrence_aggregate,
)
from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
from intergrax.runtime.diagnostics.problem_occurrence_partition_fingerprint import (
    decode_occurrence_partition_fingerprint,
    occurrence_partition_fingerprint_row_key,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-r6"
_OBSERVED_AT = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def _fingerprint_generation(store, partition_key: str) -> int | None:
    record = store.get(partition_key, occurrence_partition_fingerprint_row_key())
    if record is None:
        return None
    return decode_occurrence_partition_fingerprint(dict(record.data)).write_generation


def test_r5_crash_window_two_step_protocol_leaves_orphan_occurrence() -> None:
    """R5 non-atomic protocol: occurrence durable without fingerprint advance."""
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
        _occurrence_row_key,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
        encode_problem_occurrence_record,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    partition_key = _occurrence_partition(_TENANT, problem.problem_id)
    occurrence_id = problem_occurrence_id_for(occurrence)
    document = DocumentRecord(
        partition_key=partition_key,
        row_key=_occurrence_row_key(occurrence, occurrence_id=occurrence_id),
        data=encode_problem_occurrence_record(occurrence),
    )

    assert store.put_if_absent(document) is True
    assert _fingerprint_generation(store, partition_key) is None

    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    assert _fingerprint_generation(store, partition_key) is None


def test_r6_atomic_append_advances_fingerprint_with_occurrence() -> None:
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    partition_key = _occurrence_partition(_TENANT, problem.problem_id)

    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )
    assert _fingerprint_generation(store, partition_key) == 1
    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    assert _fingerprint_generation(store, partition_key) == 1


def test_r6_concurrent_distinct_occurrences_bump_fingerprint() -> None:
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    partition_key = _occurrence_partition(_TENANT, problem.problem_id)
    barrier = threading.Barrier(100)

    def _append(index: int) -> None:
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        barrier.wait(timeout=10)
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_append, index) for index in range(100)]
        for future in futures:
            future.result(timeout=30)

    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 100
    assert _fingerprint_generation(store, partition_key) == 100


def test_r6_concurrent_duplicate_occurrence_counts_once() -> None:
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    partition_key = _occurrence_partition(_TENANT, problem.problem_id)
    barrier = threading.Barrier(100)

    def _append() -> None:
        barrier.wait(timeout=10)
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_append) for _ in range(100)]
        for future in futures:
            future.result(timeout=30)

    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1
    assert _fingerprint_generation(store, partition_key) == 1


def test_r6_bootstrap_concurrent_append_covers_all_rows() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
        _occurrence_row_key,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
        encode_problem_occurrence_record,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem_persistence.create(problem, indexed_subject_refs=())
    partition_key = _occurrence_partition(_TENANT, problem.problem_id)

    legacy_count = 8
    for index in range(legacy_count):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        occurrence_id = problem_occurrence_id_for(occurrence)
        store.put_if_absent(
            DocumentRecord(
                partition_key=partition_key,
                row_key=_occurrence_row_key(occurrence, occurrence_id=occurrence_id),
                data=encode_problem_occurrence_record(occurrence),
            ),
        )

    barrier = threading.Barrier(2)
    bootstrap_done = threading.Event()

    def _bootstrap() -> None:
        barrier.wait(timeout=5)
        persistence.capture_occurrence_repair_boundary(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )
        bootstrap_done.set()

    def _append_new() -> None:
        barrier.wait(timeout=5)
        bootstrap_done.wait(timeout=5)
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=legacy_count),
        )[0]
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        bootstrap_future = executor.submit(_bootstrap)
        append_future = executor.submit(_append_new)
        bootstrap_future.result(timeout=10)
        append_future.result(timeout=10)

    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == legacy_count + 1
    assert _fingerprint_generation(store, partition_key) == legacy_count + 1

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=persistence,
        problem_persistence=problem_persistence,
        page_size=4,
    )
    assert repaired.occurrence_count == legacy_count + 1
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT
