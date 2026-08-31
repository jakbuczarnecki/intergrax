# © Artur Czarnecki. All rights reserved.

"""DIAG-ENTERPRISE-2-R2 durable occurrence / stats convergence proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    DocumentStoreProblemOccurrencePersistence,
    _STATS_ROW_KEY,
    wire_problem_occurrence_persistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    _sample_reconciliation_key,
    _sample_subject_ref,
    sample_occurrences,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAppendResult,
    ProblemOccurrencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.problem_occurrence_query import (
    ProblemOccurrenceQueryCursorCodec,
    _MIN_OCCURRENCE_CURSOR_SECRET_BYTES,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistenceConflictError
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_occurrence_persistence_for_tests,
    document_store_problem_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    lifecycle_engine_for_tests,
)

pytestmark = pytest.mark.unit

_TENANT = "diag-enterprise-2-r2"
_OBSERVED_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)
_VALID_SECRET = TEST_PROBLEM_LIST_CURSOR_SECRET


class _FailStatsMergeOnceStore:
    """Inject stats merge failure after durable occurrence row exists."""

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
        if (
            self._stats_failures_remaining > 0
            and document.row_key == _STATS_ROW_KEY
        ):
            self._stats_failures_remaining -= 1
            raise RuntimeError("simulated stats merge failure")
        return self._delegate.put_if_absent(document)

    def replace_if_match(self, *, expected, replacement) -> bool:
        if (
            self._stats_failures_remaining > 0
            and replacement.row_key == _STATS_ROW_KEY
        ):
            self._stats_failures_remaining -= 1
            raise RuntimeError("simulated stats merge failure")
        return self._delegate.replace_if_match(
            expected=expected,
            replacement=replacement,
        )


from intergrax.integrations.contracts.document_store import ConditionalDocumentStore

ConditionalDocumentStore.register(_FailStatsMergeOnceStore)


def test_stats_merge_failure_then_retry_converges_without_double_count() -> None:
    """Repro: durable occurrence + stats failure → ALREADY_EXISTS retry converges stats."""
    base = in_memory_document_store_for_problem_tests()
    store = _FailStatsMergeOnceStore(base)
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]

    with pytest.raises(RuntimeError, match="simulated stats merge failure"):
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )

    stats_before_retry = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats_before_retry is None

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
    assert stats.first_seen_at == occurrence.observed_at
    assert stats.last_seen_at == occurrence.observed_at


def test_s1_crash_occurrence_durable_stats_missing_retry_converges_once() -> None:
    """S1: occurrence row durable, stats missing → retry converges exactly once."""
    base = in_memory_document_store_for_problem_tests()
    store = _FailStatsMergeOnceStore(base)
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]

    with pytest.raises(RuntimeError):
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
    for _ in range(3):
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


def test_s2_crash_marker_applied_retry_already_exists_no_double_count() -> None:
    """S2: stats contribution applied, caller failed → retry leaves stats unchanged."""
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]

    assert (
        persistence.append_if_absent(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
            occurrence=occurrence,
        )
        is ProblemOccurrenceAppendResult.CREATED
    )
    stats_after_first = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats_after_first is not None
    assert stats_after_first.occurrence_count == 1

    for _ in range(5):
        assert (
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            )
            is ProblemOccurrenceAppendResult.ALREADY_EXISTS
        )
    stats_after_retry = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats_after_retry == stats_after_first


def test_s3_partial_stats_retry_converges() -> None:
    """S3: occurrence exists with missing stats row → retry converges."""
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
        _occurrence_row_key,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_id import problem_occurrence_id_for
    from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
        encode_problem_occurrence_record,
    )

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
    assert persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    ) is None

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


def test_concurrent_distinct_occurrences_count_exactly_n() -> None:
    """Two writers append different occurrences → stats count += 2."""
    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT, observed_at=_OBSERVED_AT)
    subject_a = _sample_subject_ref(tenant_id=_TENANT)
    subject_b = _sample_subject_ref(tenant_id=_TENANT)
    occurrence_a = sample_occurrences(
        subject_refs=(subject_a,),
        observed_at=_OBSERVED_AT,
    )[0]
    occurrence_b = sample_occurrences(
        subject_refs=(subject_b,),
        observed_at=_OBSERVED_AT + timedelta(minutes=1),
    )[0]
    barrier = threading.Barrier(2)
    results: list[ProblemOccurrenceAppendResult] = []

    def _append(occurrence) -> None:
        barrier.wait(timeout=5)
        results.append(
            persistence.append_if_absent(
                tenant_id=_TENANT,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_append, occurrence_a),
            executor.submit(_append, occurrence_b),
        ]
        for future in futures:
            future.result(timeout=10)

    assert all(
        result is ProblemOccurrenceAppendResult.CREATED for result in results
    )
    stats = persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 2
    assert stats.first_seen_at == occurrence_a.observed_at
    assert stats.last_seen_at == occurrence_b.observed_at


def test_malformed_stats_record_fails_closed() -> None:
    from intergrax.integrations.contracts.document_store import DocumentRecord
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _occurrence_partition,
    )
    from intergrax.runtime.diagnostics.problem_occurrence_record_codec import (
        encode_occurrence_stats_record,
    )

    store = in_memory_document_store_for_problem_tests()
    persistence = document_store_occurrence_persistence_for_tests(store)
    problem = sample_problem(tenant_id=_TENANT)
    partition = _occurrence_partition(_TENANT, problem.problem_id)
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=_STATS_ROW_KEY,
            data=encode_occurrence_stats_record(
                occurrence_count=2,
                first_seen_at=_OBSERVED_AT + timedelta(hours=1),
                last_seen_at=_OBSERVED_AT,
            ),
        ),
    )
    with pytest.raises(ProblemOccurrencePersistenceIntegrityError):
        persistence.aggregate_stats(
            tenant_id=_TENANT,
            problem_id=problem.problem_id,
        )


@pytest.mark.parametrize(
    ("observed_at", "expected_micros"),
    [
        (datetime(1970, 1, 1, 0, 0, 0, 123456, tzinfo=UTC), 123_456),
        (
            datetime(2026, 8, 31, 9, 0, tzinfo=timezone(timedelta(hours=2))),
            int(datetime(2026, 8, 31, 7, 0, tzinfo=UTC).timestamp() * 1_000_000),
        ),
        (
            datetime(2026, 8, 31, 9, 0, tzinfo=timezone(timedelta(hours=2))),
            int(datetime(2026, 8, 31, 7, 0, tzinfo=UTC).timestamp() * 1_000_000),
        ),
    ],
)
def test_observed_at_micros_integer_only_no_float_drift(
    observed_at: datetime,
    expected_micros: int,
) -> None:
    from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
        _observed_at_micros,
    )

    assert _observed_at_micros(observed_at) == expected_micros


def test_occurrence_cursor_secret_validation() -> None:
    store = in_memory_document_store_for_problem_tests()
    with pytest.raises(ValueError, match="problem_occurrence_cursor_secret_invalid"):
        DocumentStoreProblemOccurrencePersistence(
            store,
            occurrence_cursor_secret=b"",
        )
    with pytest.raises(ValueError, match="problem_occurrence_cursor_secret_too_short"):
        DocumentStoreProblemOccurrencePersistence(
            store,
            occurrence_cursor_secret=b"x" * 31,
        )
    DocumentStoreProblemOccurrencePersistence(
        store,
        occurrence_cursor_secret=b"x" * 32,
    )
    with pytest.raises(ValueError, match="problem_occurrence_cursor_secret_too_short"):
        ProblemOccurrenceQueryCursorCodec(secret=b"y" * 31)
    assert len(_VALID_SECRET) >= _MIN_OCCURRENCE_CURSOR_SECRET_BYTES
    with pytest.raises(ValueError, match="problem_occurrence_cursor_secret_too_short"):
        wire_problem_occurrence_persistence(
            document_store=store,
            occurrence_cursor_secret=b"z" * 31,
        )


class _FailOccurrenceAppendPersistence:
    def __init__(self, delegate) -> None:
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


def test_f2_occurrence_append_fails_before_durable_row_retry_succeeds_once() -> None:
    """F2: append fails before durable row → aggregate not advanced → retry succeeds."""
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

    store = InMemoryDocumentStore()
    problem_persistence = document_store_problem_persistence_for_tests(store)
    delegate_occurrence = document_store_occurrence_persistence_for_tests(store)
    failing_occurrence = _FailOccurrenceAppendPersistence(delegate_occurrence)
    engine = lifecycle_engine_for_tests(
        problem_persistence,
        failing_occurrence,
        document_store=store,
    )
    from intergrax.runtime.diagnostics.persistence_conformance import (
        _sample_reconciliation_key,
        _sample_signature,
    )
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

    subject = _sample_subject_ref(tenant_id=_TENANT)
    reconciliation_key = _sample_reconciliation_key(tenant_id=_TENANT)
    baseline = sample_problem(
        tenant_id=_TENANT,
        subject_refs=(subject,),
        reconciliation_key=reconciliation_key,
        observed_at=_OBSERVED_AT,
        occurrence_count=1,
    )
    problem_persistence.create(baseline, indexed_subject_refs=(subject,))
    new_subject = _sample_subject_ref(tenant_id=_TENANT)
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
    observed_at = _OBSERVED_AT + timedelta(minutes=5)

    with pytest.raises(RuntimeError, match="simulated occurrence append failure"):
        engine.reconcile(grouping, observed_at=observed_at)

    before_retry = problem_persistence.get(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert before_retry is not None
    assert before_retry.occurrence_count == 1
    stats_before = delegate_occurrence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert stats_before is None or stats_before.occurrence_count <= 1

    engine = lifecycle_engine_for_tests(
        problem_persistence,
        delegate_occurrence,
        document_store=store,
    )
    result = engine.reconcile(grouping, observed_at=observed_at)
    final = problem_persistence.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    stats = delegate_occurrence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1
    assert final is not None
    assert final.occurrence_count == 2
    assert len(result.updated) == 1


def test_f3_restart_persistence_object_converges_stats() -> None:
    """F3: durable occurrence + stats failure → new persistence instance → convergence."""
    base = in_memory_document_store_for_problem_tests()
    store = _FailStatsMergeOnceStore(base)
    persistence = document_store_occurrence_persistence_for_tests(store)
    subject = _sample_subject_ref(tenant_id=_TENANT)
    problem = sample_problem(tenant_id=_TENANT, subject_refs=(subject,), observed_at=_OBSERVED_AT)
    occurrence = sample_occurrences(subject_refs=(subject,), observed_at=_OBSERVED_AT)[0]

    with pytest.raises(RuntimeError):
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
        is ProblemOccurrenceAppendResult.ALREADY_EXISTS
    )
    stats = restarted.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=problem.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 1


class _FailAfterOccurrenceAppendPersistence:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
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
        if not self._failed:
            self._failed = True
            raise ProblemPersistenceConflictError("simulated aggregate update failure")
        return self._delegate.update(
            record,
            expected_version=expected_version,
            indexed_subject_refs=indexed_subject_refs,
        )

    def close(self) -> None:
        self._delegate.close()


def test_f1_occurrence_stats_correct_problem_update_fails_retry_converges() -> None:
    """F1: occurrence+stats correct, Problem update fails → retry → exact count."""
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore

    store = InMemoryDocumentStore()
    delegate = document_store_problem_persistence_for_tests(store)
    occurrence_persistence = document_store_occurrence_persistence_for_tests(store)
    failing = _FailAfterOccurrenceAppendPersistence(delegate)

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

    stats = occurrence_persistence.aggregate_stats(
        tenant_id=_TENANT,
        problem_id=baseline.problem_id,
    )
    assert stats is not None
    assert stats.occurrence_count == 2

    from intergrax.runtime.diagnostics.problem_occurrence_aggregate_convergence import (
        converge_problem_from_durable_stats,
    )

    loaded = delegate.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    assert loaded is not None
    converged = converge_problem_from_durable_stats(loaded, stats=stats)
    delegate.update(
        converged,
        expected_version=loaded.record_version,
        indexed_subject_refs=(new_subject,),
    )
    final = delegate.get(tenant_id=_TENANT, problem_id=baseline.problem_id)
    assert final is not None
    assert final.occurrence_count == 2
