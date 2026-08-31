# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R4 paginated aggregate reconciliation proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore, DocumentRecord
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_signature,
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleEngine,
    ProblemLifecycleIntegrityError,
)
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    DEFAULT_REPAIR_PAGE_SIZE,
    ProblemOccurrenceAggregateHealth,
    mark_problem_reconciliation_required,
    reconcile_problem_occurrence_aggregate,
    scan_occurrence_aggregate,
)
from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
)
from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
    encode_problem_occurrence_record,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceConflictError
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    _occurrence_partition,
    _occurrence_row_key,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_lifecycle_stack_for_tests,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    lifecycle_engine_for_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-r4"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)


class _FailProblemUpdateOnce:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._remaining = 1

    def get(self, *, tenant_id: str, problem_id):
        return self._delegate.get(tenant_id=tenant_id, problem_id=problem_id)

    def query_problems(self, *, tenant_id: str, status=None, limit: int, cursor=None):
        return self._delegate.query_problems(
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            cursor=cursor,
        )

    def find_by_reconciliation_key(self, *, tenant_id: str, reconciliation_key):
        return self._delegate.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        )

    def find_by_subject_ref(self, *, tenant_id: str, subject_ref):
        return self._delegate.find_by_subject_ref(
            tenant_id=tenant_id,
            subject_ref=subject_ref,
        )

    def create(self, record: Problem, *, indexed_subject_refs=()):
        return self._delegate.create(record, indexed_subject_refs=indexed_subject_refs)

    def update(self, record: Problem, *, expected_version: int, indexed_subject_refs=()):
        if self._remaining > 0 and record.occurrence_count > 1:
            self._remaining -= 1
            raise ProblemPersistenceConflictError("simulated aggregate update failure")
        return self._delegate.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self._delegate.close()


ConditionalDocumentStore.register(_FailProblemUpdateOnce)


def _seed_n_occurrences(persistence, *, problem, count: int) -> None:
    for index in range(count):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )


def _grouping_for_subjects(*subjects):
    from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
        STRATEGY_ID,
        STRATEGY_VERSION,
    )
    from intergrax.runtime.diagnostics.problem_grouping import (
        DeterministicProblemGroupingBasis,
        ProblemGroupingCandidate,
        ProblemGroupingMethod,
        ProblemGroupingProvenance,
        ProblemGroupingResult,
    )

    basis = DeterministicProblemGroupingBasis(signature=_sample_signature())
    candidate = ProblemGroupingCandidate(
        members=subjects,
        provenance=ProblemGroupingProvenance(
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
            supporting_subject_refs=subjects,
            basis=basis,
        ),
    )
    return ProblemGroupingResult(
        tenant_id=_TENANT,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        candidates=(candidate,),
        ungrouped_subjects=(),
    )


def test_r3_blocker_duplicate_retry_after_two_appends_stays_at_101() -> None:
    """R3 race analog: A+B durable rows, retry A must not reach 102."""
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT, occurrence_count=99)
    _seed_n_occurrences(persistence, problem=problem, count=99)

    occurrence_a = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=100),
    )[0]
    occurrence_b = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=101),
    )[0]
    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence_a,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )
    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence_b,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )

    problem_persistence.create(problem, indexed_subject_refs=())
    updated = Problem(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=occurrence_b.observed_at,
        occurrence_count=101,
        provenance=problem.provenance,
        record_version=2,
    )
    problem_persistence.update(updated, expected_version=1)

    for _ in range(5):
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence_a,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )

    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 101
    stored = problem_persistence.get(tenant_id=_TENANT, problem_id=problem.problem_id)
    assert stored is not None
    assert stored.occurrence_count == 101


def test_a3_occurrence_saved_aggregate_fail_marks_repair_then_converges() -> None:
    store = in_memory_document_store_for_problem_tests()
    delegate = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    failing = _FailProblemUpdateOnce(delegate)
    engine = lifecycle_engine_for_tests(failing, occurrence_persistence, document_store=store)

    subject = _sample_subject_ref(tenant_id=_TENANT)
    baseline = sample_problem(
        tenant_id=_TENANT,
        subject_refs=(subject,),
        reconciliation_key=_sample_reconciliation_key(tenant_id=_TENANT),
        observed_at=_OBSERVED_AT,
    )
    failing.create(baseline, indexed_subject_refs=(subject,))
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
        occurrence=sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0],
    )

    new_subject = _sample_subject_ref(tenant_id=_TENANT)
    grouping = _grouping_for_subjects(subject, new_subject)
    observed_at = _OBSERVED_AT + timedelta(minutes=1)
    engine.reconcile(grouping, observed_at=observed_at)

    stored = delegate.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    assert stored is not None
    assert stored.occurrence_count == 2
    assert stored.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT
    scan = scan_occurrence_aggregate(
        occurrence_persistence,
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert scan.occurrence_count == 2


def test_repair_paginated_exact_100k() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())

    for index in range(100_000):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(seconds=index),
        )[0]
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=occurrence_persistence,
        problem_persistence=problem_persistence,
        page_size=500,
    )
    assert repaired.occurrence_count == 100_000
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT
    assert repaired.first_seen_at == _OBSERVED_AT
    assert repaired.last_seen_at == _OBSERVED_AT + timedelta(seconds=99_999)


@pytest.mark.no_ci
def test_repair_paginated_exact_1m() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())

    for index in range(1_000_000):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(
            subject_refs=(subject,),
            observed_at=_OBSERVED_AT + timedelta(microseconds=index),
        )[0]
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=occurrence_persistence,
        problem_persistence=problem_persistence,
        page_size=DEFAULT_REPAIR_PAGE_SIZE,
    )
    assert repaired.occurrence_count == 1_000_000


def test_concurrent_duplicate_append_single_row() -> None:
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
    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1


def test_a6_new_occurrence_during_repair_not_lost() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())

    seeded = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT,
    )[0]
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence=seeded,
    )
    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)

    extra = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(minutes=1),
    )[0]
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence=extra,
    )
    hot = Problem(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=seeded.observed_at,
        last_seen_at=extra.observed_at,
        occurrence_count=2,
        provenance=problem.provenance,
        record_version=3,
    )
    problem_persistence.update(hot, expected_version=2)

    repaired = reconcile_problem_occurrence_aggregate(
        hot,
        occurrence_persistence=occurrence_persistence,
        problem_persistence=problem_persistence,
    )
    assert repaired.occurrence_count == 2


def test_manual_occurrence_seed_without_central_stats() -> None:
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT,
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
    scan = scan_occurrence_aggregate(
        persistence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1


def test_empty_history_repair_fails_closed() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    with pytest.raises(ProblemLifecycleIntegrityError):
        reconcile_problem_occurrence_aggregate(
            stale,
            occurrence_persistence=occurrence_persistence,
            problem_persistence=problem_persistence,
        )


def test_lifecycle_engine_reconcile_occurrence_aggregate() -> None:
    _, problem_persistence, occurrence_persistence, engine = document_store_lifecycle_stack_for_tests()
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    occurrence = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT,
    )[0]
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
        occurrence=occurrence,
    )
    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    repaired = engine.reconcile_occurrence_aggregate(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert repaired.occurrence_count == 1
