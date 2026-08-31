# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R5 snapshot-safe occurrence aggregate repair proofs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleIntegrityError,
    ProblemOccurrenceAggregateHealth,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
    DEFAULT_REPAIR_PAGE_SIZE,
    mark_problem_reconciliation_required,
    reconcile_problem_occurrence_aggregate,
    scan_occurrence_aggregate,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePage,
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_occurrence_partition_fingerprint import (
    ProblemOccurrenceRepairBoundary,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceConflictError
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-r5"
_OBSERVED_AT = datetime(2026, 8, 31, 10, 0, tzinfo=UTC)


def _seed_n_occurrences(
    persistence: ProblemOccurrencePersistence,
    *,
    problem: Problem,
    count: int,
    start_index: int = 0,
) -> None:
    for index in range(start_index, start_index + count):
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


class _LateInsertDuringScanPersistence(ProblemOccurrencePersistence):
    """Inserts a late occurrence after a bounded number of scanned rows."""

    def __init__(
        self,
        delegate: ProblemOccurrencePersistence,
        *,
        trigger_after_items: int,
        late_occurrence: object,
        tenant_id: str,
        problem_id: object,
        on_trigger: Callable[[], None] | None = None,
    ) -> None:
        self._delegate = delegate
        self._trigger_after_items = trigger_after_items
        self._late_occurrence = late_occurrence
        self._tenant_id = tenant_id
        self._problem_id = problem_id
        self._on_trigger = on_trigger
        self._items_returned = 0
        self._triggered = False
        self._lock = threading.Lock()

    def append_if_absent(self, **kwargs):
        return self._delegate.append_if_absent(**kwargs)

    def capture_occurrence_repair_boundary(self, **kwargs):
        return self._delegate.capture_occurrence_repair_boundary(**kwargs)

    def query_occurrences(self, **kwargs) -> ProblemOccurrencePage:
        page = self._delegate.query_occurrences(**kwargs)
        with self._lock:
            self._items_returned += len(page.items)
            if (
                not self._triggered
                and self._items_returned >= self._trigger_after_items
                and page.items
            ):
                self._triggered = True
                self._delegate.append_if_absent(
                    tenant_id=self._tenant_id,
                    problem_id=self._problem_id,
                    occurrence=self._late_occurrence,
                )
                if self._on_trigger is not None:
                    self._on_trigger()
        return page


class _UnstableBoundaryPersistence(ProblemOccurrencePersistence):
    """Forces repair fingerprint instability on every boundary capture."""

    def __init__(self, delegate: ProblemOccurrencePersistence) -> None:
        self._delegate = delegate
        self._capture_calls = 0

    def append_if_absent(self, **kwargs):
        return self._delegate.append_if_absent(**kwargs)

    def capture_occurrence_repair_boundary(self, **kwargs):
        base = self._delegate.capture_occurrence_repair_boundary(**kwargs)
        if base is None:
            return None
        self._capture_calls += 1
        return ProblemOccurrenceRepairBoundary(
            write_generation=self._capture_calls,
            min_row_key=base.min_row_key,
            terminal_row_key=base.terminal_row_key,
        )

    def query_occurrences(self, **kwargs) -> ProblemOccurrencePage:
        return self._delegate.query_occurrences(**kwargs)


class _FailProblemUpdatePersistence:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self._fail_remaining = 0

    def arm_failures(self, count: int) -> None:
        self._fail_remaining = count

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
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise ProblemPersistenceConflictError("simulated aggregate update failure")
        return self._delegate.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self._delegate.close()


def test_r4_late_insert_before_cursor_cannot_false_consistent() -> None:
    """R4 blocker: paginated scan may miss late insert; R5 must not write false CONSISTENT."""
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(
        tenant_id=_TENANT,
        occurrence_count=1000,
        observed_at=_OBSERVED_AT,
    )
    _seed_n_occurrences(occurrence_persistence, problem=problem, count=1001)

    stale = Problem(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=_OBSERVED_AT + timedelta(seconds=999),
        occurrence_count=1000,
        provenance=problem.provenance,
        record_version=1,
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.RECONCILIATION_REQUIRED,
    )
    problem_persistence.create(stale, indexed_subject_refs=())

    late_occurrence = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=999, microseconds=500000),
    )[0]
    intercepting = _LateInsertDuringScanPersistence(
        occurrence_persistence,
        trigger_after_items=500,
        late_occurrence=late_occurrence,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    failing = _FailProblemUpdatePersistence(problem_persistence)
    failing.arm_failures(1)

    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=intercepting,
        problem_persistence=failing,
        page_size=500,
    )
    assert repaired.occurrence_count == 1002
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT
    scan = scan_occurrence_aggregate(
        intercepting,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert scan.occurrence_count == 1002


def test_r5_c1_late_insert_before_cursor_position() -> None:
    store = in_memory_document_store_for_problem_tests()
    base_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    _seed_n_occurrences(base_persistence, problem=problem, count=1200)

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    late = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=1199, microseconds=500000),
    )[0]
    intercepting = _LateInsertDuringScanPersistence(
        base_persistence,
        trigger_after_items=400,
        late_occurrence=late,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=intercepting,
        problem_persistence=problem_persistence,
        page_size=400,
    )
    assert repaired.occurrence_count == 1201
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


def test_r5_c2_late_insert_after_cursor_position() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    _seed_n_occurrences(occurrence_persistence, problem=problem, count=900)

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    late = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=50),
    )[0]
    intercepting = _LateInsertDuringScanPersistence(
        occurrence_persistence,
        trigger_after_items=500,
        late_occurrence=late,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=intercepting,
        problem_persistence=problem_persistence,
        page_size=500,
    )
    assert repaired.occurrence_count == 901
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


def test_r5_c3_same_timestamp_tie_break_insert() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    shared_time = _OBSERVED_AT + timedelta(seconds=42)
    for _ in range(600):
        subject = _sample_subject_ref(tenant_id=_TENANT)
        occurrence = sample_occurrences(subject_refs=(subject,), observed_at=shared_time)[0]
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    late = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=shared_time,
    )[0]
    intercepting = _LateInsertDuringScanPersistence(
        occurrence_persistence,
        trigger_after_items=300,
        late_occurrence=late,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=intercepting,
        problem_persistence=problem_persistence,
        page_size=300,
    )
    assert repaired.occurrence_count == 601
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


def test_r5_c4_durable_occurrence_problem_update_fail_during_scan() -> None:
    store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    delegate = document_store_problem_persistence_for_tests(store)
    failing = _FailProblemUpdatePersistence(delegate)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    delegate.create(problem, indexed_subject_refs=())
    _seed_n_occurrences(occurrence_persistence, problem=problem, count=10)
    stale = mark_problem_reconciliation_required(problem)
    delegate.update(stale, expected_version=1)
    failing.arm_failures(2)

    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=occurrence_persistence,
        problem_persistence=failing,
    )
    assert repaired.occurrence_count == 10
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


def test_r5_c5_continuous_writers_bounded_termination() -> None:
    store = in_memory_document_store_for_problem_tests()
    base_persistence = document_store_occurrence_persistence_for_tests(store)
    unstable = _UnstableBoundaryPersistence(base_persistence)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    _seed_n_occurrences(base_persistence, problem=problem, count=50)

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)

    with pytest.raises(ProblemLifecycleIntegrityError):
        reconcile_problem_occurrence_aggregate(
            stale,
            occurrence_persistence=unstable,
            problem_persistence=problem_persistence,
            page_size=25,
        )

    stored = problem_persistence.get(tenant_id=_TENANT, problem_id=problem.problem_id)
    assert stored is not None
    assert (
        stored.occurrence_aggregate_health
        is ProblemOccurrenceAggregateHealth.RECONCILIATION_REQUIRED
    )


def test_repair_paginated_exact_100k_with_late_insert() -> None:
    store = in_memory_document_store_for_problem_tests()
    base_persistence = document_store_occurrence_persistence_for_tests(store)
    problem_persistence = document_store_problem_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, occurrence_count=0, observed_at=_OBSERVED_AT)
    problem_persistence.create(problem, indexed_subject_refs=())
    _seed_n_occurrences(base_persistence, problem=problem, count=100_000)

    stale = mark_problem_reconciliation_required(problem)
    problem_persistence.update(stale, expected_version=1)
    late = sample_occurrences(
        subject_refs=(_sample_subject_ref(tenant_id=_TENANT),),
        observed_at=_OBSERVED_AT + timedelta(seconds=99_999, microseconds=500000),
    )[0]
    intercepting = _LateInsertDuringScanPersistence(
        base_persistence,
        trigger_after_items=500,
        late_occurrence=late,
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    repaired = reconcile_problem_occurrence_aggregate(
        stale,
        occurrence_persistence=intercepting,
        problem_persistence=problem_persistence,
        page_size=500,
    )
    assert repaired.occurrence_count == 100_001
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT


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
    assert repaired.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.CONSISTENT
