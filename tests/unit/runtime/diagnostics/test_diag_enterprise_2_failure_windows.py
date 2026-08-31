# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2 write-protocol failure windows F1–F6."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_signature,
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemLifecycleEngine
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceConflictError
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    lifecycle_engine_for_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-failure-windows"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)


class _FailAfterOccurrenceAppendPersistence:
    def __init__(
        self,
        delegate: ProblemPersistence,
        *,
        fail_once: bool = True,
    ) -> None:
        self._delegate = delegate
        self._fail_once = fail_once
        self._failed = False

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
        if self._fail_once and not self._failed:
            self._failed = True
            raise ProblemPersistenceConflictError("simulated aggregate update failure")
        return self._delegate.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self._delegate.close()


class _FailOccurrenceAppendPersistence:
    def __init__(self, delegate: ProblemOccurrencePersistence) -> None:
        self._delegate = delegate
        self._failed_once = False

    def append_if_absent(self, *, tenant_id: str, problem_id, occurrence):
        if not self._failed_once:
            self._failed_once = True
            raise RuntimeError("simulated occurrence append failure")
        return self._delegate.append_if_absent(
            tenant_id=tenant_id,
            problem_id=problem_id,
            occurrence=occurrence,
        )

    def query_occurrences(self, *, tenant_id: str, problem_id, limit: int, cursor=None):
        return self._delegate.query_occurrences(
            tenant_id=tenant_id,
            problem_id=problem_id,
            limit=limit,
            cursor=cursor,
        )

    def aggregate_stats(self, *, tenant_id: str, problem_id):
        return self._delegate.aggregate_stats(
            tenant_id=tenant_id,
            problem_id=problem_id,
        )


def _lifecycle_stack() -> tuple[InMemoryDocumentStore, ProblemLifecycleEngine, ProblemPersistence]:
    store = InMemoryDocumentStore()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    engine = lifecycle_engine_for_tests(
        problem_persistence,
        occurrence_persistence,
        document_store=store,
    )
    return store, engine, problem_persistence


def test_f1_occurrence_success_aggregate_fail_converges_on_retry() -> None:
    """F1: durable occurrence survives aggregate CAS failure; retry converges count."""
    store = InMemoryDocumentStore()
    delegate = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    failing = _FailAfterOccurrenceAppendPersistence(delegate)
    engine = ProblemLifecycleEngine(failing, occurrence_persistence)

    subject = _sample_subject_ref(tenant_id=_TENANT)
    reconciliation_key = _sample_reconciliation_key(tenant_id=_TENANT)
    baseline = sample_problem(
        tenant_id=_TENANT,
        subject_refs=(subject,),
        reconciliation_key=reconciliation_key,
        observed_at=_OBSERVED_AT,
    )
    failing.create(baseline, indexed_subject_refs=(subject,))
    baseline_occurrence = sample_occurrences(
        subject_refs=(subject,),
        observed_at=_OBSERVED_AT,
    )[0]
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
        occurrence=baseline_occurrence,
    )

    new_subject = _sample_subject_ref(tenant_id=_TENANT)
    occurrence = sample_occurrences(
        subject_refs=(new_subject,),
        observed_at=_OBSERVED_AT + timedelta(minutes=1),
    )[0]
    occurrence_persistence.append_if_absent(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
        occurrence=occurrence,
    )

    with pytest.raises(ProblemPersistenceConflictError):
        failing.update(
            Problem(
                problem_id=baseline.problem_id,
                tenant_id=baseline.tenant_id,
                status=baseline.status,
                first_seen_at=baseline.first_seen_at,
                last_seen_at=occurrence.observed_at,
                occurrence_count=2,
                provenance=baseline.provenance,
                record_version=2,
            ),
            expected_version=1,
            indexed_subject_refs=(new_subject,),
        )

  # Retry path via lifecycle convergence
    from intergrax.runtime.diagnostics.problem_grouping import (
        DeterministicProblemGroupingBasis,
        ProblemGroupingCandidate,
        ProblemGroupingMethod,
        ProblemGroupingProvenance,
        ProblemGroupingResult,
    )
    from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
        STRATEGY_ID,
        STRATEGY_VERSION,
    )

    basis = DeterministicProblemGroupingBasis(signature=_sample_signature())
    candidate = ProblemGroupingCandidate(
        members=(new_subject,),
        provenance=ProblemGroupingProvenance(
            strategy_id=STRATEGY_ID,
            strategy_version=STRATEGY_VERSION,
            method=ProblemGroupingMethod.DETERMINISTIC,
            supporting_subject_refs=(new_subject,),
            basis=basis,
        ),
    )
    grouping = ProblemGroupingResult(
        tenant_id=_TENANT,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        candidates=(candidate,),
        ungrouped_subjects=(),
    )
    result = engine.reconcile(grouping, observed_at=occurrence.observed_at)
    final = delegate.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 2
    assert final is not None
    assert final.occurrence_count == 2
    assert len(result.updated) == 1


def test_f4_duplicate_append_does_not_double_count() -> None:
    """F4: duplicate durable append leaves aggregate count stable."""
    store = InMemoryDocumentStore()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    assert (
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )
    assert (
        occurrence_persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


def test_f5_concurrent_duplicate_append_counts_once() -> None:
    """F5: concurrent duplicate durable appends converge to one row."""
    store = InMemoryDocumentStore()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]
    barrier = threading.Barrier(2)
    results: list[ProblemOccurrenceAppendResult] = []

    def _append() -> None:
        barrier.wait(timeout=5)
        results.append(
            occurrence_persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_append), executor.submit(_append)]
        for future in futures:
            future.result(timeout=10)
    assert sorted(results, key=str) == sorted(
        [
            ProblemOccurrenceAppendResult.CREATED,
            ProblemOccurrenceAppendResult.ALREADY_EXISTS,
        ],
        key=str,
    )
    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1
