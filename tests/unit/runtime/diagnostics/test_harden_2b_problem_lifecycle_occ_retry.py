# © Artur Czarnecki. All rights reserved.

"""HARDEN-2B — bounded lifecycle OCC retry proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
)
from intergrax.runtime.diagnostics.persistence_conformance import (
    query_all_problems_for_tenant,
    _sample_reconciliation_key,
    _sample_signature,
    _sample_subject_ref,
    sample_problem,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemGroupingBasis,
    DeterministicProblemSignature,
    ProblemGroupingCandidate,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingResult,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemLifecycleEngine,
    ProblemLifecycleIntegrityError,
    ProblemLifecycleResult,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
    ProblemPersistenceIntegrityReason,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    TEST_PROBLEM_LIST_CURSOR_SECRET,
    document_store_problem_persistence_for_tests,
)

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
_OBSERVED_AT_A = _OBSERVED_AT + timedelta(minutes=1)
_OBSERVED_AT_B = _OBSERVED_AT + timedelta(minutes=2)
_RESOLVED_AT = _OBSERVED_AT + timedelta(hours=1)


def _singleton_grouping_result(
    *,
    tenant_id: str,
    member,
    signature: DeterministicProblemSignature,
    observed_at: datetime,
) -> ProblemGroupingResult:
    del observed_at  # reconcile() owns observed_at; candidate members carry invocation time.
    basis = DeterministicProblemGroupingBasis(signature=signature)
    provenance = ProblemGroupingProvenance(
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        supporting_subject_refs=(member,),
        basis=basis,
    )
    candidate = ProblemGroupingCandidate(members=(member,), provenance=provenance)
    return ProblemGroupingResult(
        tenant_id=tenant_id,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        candidates=(candidate,),
        ungrouped_subjects=(),
    )


class _SynchronizedUpdatePersistence(DocumentStoreProblemPersistence):
    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        *,
        update_barrier: threading.Barrier,
        synchronized_expected_version: int,
    ) -> None:
        super().__init__(
            document_store,
            list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
            document_query_cursor_codec=document_store.query_cursor_codec,
        )
        self._update_barrier = update_barrier
        self._synchronized_expected_version = synchronized_expected_version

    def update(self, record: Problem, *, expected_version: int) -> Problem:
        if expected_version == self._synchronized_expected_version:
            self._update_barrier.wait(timeout=5)
        return super().update(record, expected_version=expected_version)


class _SynchronizedResolvePersistence(DocumentStoreProblemPersistence):
    def __init__(
        self,
        document_store: InMemoryDocumentStore,
        *,
        resolve_barrier: threading.Barrier,
        synchronized_expected_version: int,
    ) -> None:
        super().__init__(
            document_store,
            list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
            document_query_cursor_codec=document_store.query_cursor_codec,
        )
        self._resolve_barrier = resolve_barrier
        self._synchronized_expected_version = synchronized_expected_version

    def update(self, record: Problem, *, expected_version: int) -> Problem:
        if (
            record.status is ProblemStatus.RESOLVED
            and expected_version == self._synchronized_expected_version
        ):
            self._resolve_barrier.wait(timeout=5)
        return super().update(record, expected_version=expected_version)


class _AlwaysConflictReplaceStore(InMemoryDocumentStore):
    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        del expected, replacement
        return False


class _ConflictThenVanishPersistence:
    """Persistence double: first update conflicts, reload get returns None."""

    def __init__(self, delegate: ProblemPersistence) -> None:
        self._delegate = delegate
        self._force_conflict_once = True
        self._vanish_on_next_get = False

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> Problem | None:
        if self._vanish_on_next_get:
            self._vanish_on_next_get = False
            return None
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

    def create(self, record: Problem) -> Problem:
        return self._delegate.create(record)

    def update(self, record: Problem, *, expected_version: int) -> Problem:
        if self._force_conflict_once:
            self._force_conflict_once = False
            self._vanish_on_next_get = True
            raise ProblemPersistenceConflictError("forced conflict for reload-vanish proof")
        return self._delegate.update(record, expected_version=expected_version)

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()


class _CreateConflictLookupIntegrityPersistence:
    """Persistence double: create conflicts; lookup raises integrity failure on convergence."""

    def __init__(
        self,
        *,
        lookup_exc: ProblemPersistenceIntegrityError,
    ) -> None:
        self._lookup_exc = lookup_exc
        self._lookup_calls = 0

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> Problem | None:
        return None

    def query_problems(self, *, tenant_id: str, status=None, limit: int, cursor=None):
        from intergrax.runtime.diagnostics.problem_list_query import ProblemListPage

        return ProblemListPage(problems=(), next_cursor=None, has_more=False)

    def find_by_reconciliation_key(self, *, tenant_id: str, reconciliation_key):
        self._lookup_calls += 1
        if self._lookup_calls == 1:
            return None
        raise self._lookup_exc

    def find_by_subject_ref(self, *, tenant_id: str, subject_ref):
        return None

    def create(self, record: Problem) -> Problem:
        raise ProblemPersistenceConflictError("forced create conflict for lookup proof")

    def update(self, record: Problem, *, expected_version: int) -> Problem:
        raise ProblemPersistenceConflictError("unexpected update in lookup proof")

    def close(self) -> None:
        return None


def test_harden_2d_create_lookup_integrity_error_without_pending_reason_fails_closed() -> None:
    tenant_id = "harden-2d-create-lookup-integrity"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    subject = _sample_subject_ref(tenant_id=tenant_id)
    grouping = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject,
        signature=signature,
        observed_at=_OBSERVED_AT_A,
    )

    lookup_cases = [
        ProblemPersistenceIntegrityError("orphan reconciliation index"),
        ProblemPersistenceIntegrityError("integrity failure without typed reason"),
    ]
    other_reason_exc = ProblemPersistenceIntegrityError("integrity failure with other reason")
    other_reason_exc.reason = object()
    lookup_cases.append(other_reason_exc)

    for lookup_exc in lookup_cases:
        persistence = _CreateConflictLookupIntegrityPersistence(lookup_exc=lookup_exc)
        lifecycle = ProblemLifecycleEngine(persistence)
        with pytest.raises(
            ProblemLifecycleIntegrityError,
            match="persistence lookup failure",
        ) as exc_info:
            lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A)
        assert isinstance(exc_info.value.__cause__, ProblemPersistenceIntegrityError)
        persistence.close()


def test_harden_2b_lifecycle_update_race_preserves_distinct_occurrences() -> None:
    """
    HARDEN-2B lifecycle proof:

    baseline occurrence_count=1; two lifecycle engines add different occurrences.
    """
    tenant_id = "harden-2b-lifecycle-update-tenant"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )
    store = InMemoryDocumentStore()
    update_barrier = threading.Barrier(2)
    assert baseline.occurrence_count == 1
    assert baseline.record_version == 1

    seed = document_store_problem_persistence_for_tests(store)
    seed.create(baseline)
    seed.close()

    grouping_a = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject_a,
        signature=signature,
        observed_at=_OBSERVED_AT_A,
    )
    grouping_b = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject_b,
        signature=signature,
        observed_at=_OBSERVED_AT_B,
    )

    results: list[ProblemLifecycleResult] = []
    errors: list[BaseException] = []

    def _reconcile(grouping_result: ProblemGroupingResult, observed_at: datetime) -> None:
        persistence = _SynchronizedUpdatePersistence(
            store,
            update_barrier=update_barrier,
            synchronized_expected_version=baseline.record_version,
        )
        lifecycle = ProblemLifecycleEngine(persistence)
        try:
            results.append(
                lifecycle.reconcile(grouping_result, observed_at=observed_at),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_reconcile, grouping_a, _OBSERVED_AT_A),
            executor.submit(_reconcile, grouping_b, _OBSERVED_AT_B),
        ]
        for future in futures:
            future.result(timeout=10)

    assert errors == []
    assert len(results) == 2
    assert all(len(result.updated) == 1 for result in results)
    assert all(result.created == () for result in results)
    assert all(result.unchanged == () for result in results)

    verifier = document_store_problem_persistence_for_tests(store)
    try:
        final = verifier.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final is not None
        assert final.record_version == 3
        assert final.occurrence_count == 3
        assert len(final.occurrences) == 3
        final_subjects = {occurrence.subject_ref for occurrence in final.occurrences}
        assert final_subjects == {baseline_subject, subject_a, subject_b}
        assert verifier.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        ) == final
    finally:
        verifier.close()


def test_harden_2b_retry_exhaustion_raises_lifecycle_integrity_error() -> None:
    store = _AlwaysConflictReplaceStore()
    tenant_id = "harden-2b-retry-exhaustion"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    new_subject = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )

    persistence = document_store_problem_persistence_for_tests(store)
    lifecycle = ProblemLifecycleEngine(persistence)
    try:
        persistence.create(baseline)
        grouping = _singleton_grouping_result(
            tenant_id=tenant_id,
            member=new_subject,
            signature=signature,
            observed_at=_OBSERVED_AT_A,
        )
        with pytest.raises(ProblemLifecycleIntegrityError) as exc_info:
            lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A)
        assert isinstance(exc_info.value.__cause__, ProblemPersistenceConflictError)
        final = persistence.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final == baseline
    finally:
        persistence.close()


def test_harden_2b_reload_disappears_after_conflict_fails_closed() -> None:
    from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
        InMemoryProblemPersistence,
    )

    tenant_id = "harden-2b-reload-vanish"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    new_subject = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )

    delegate = InMemoryProblemPersistence()
    persistence = _ConflictThenVanishPersistence(delegate)
    lifecycle = ProblemLifecycleEngine(persistence)
    persistence.create(baseline)
    grouping = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=new_subject,
        signature=signature,
        observed_at=_OBSERVED_AT_A,
    )
    with pytest.raises(ProblemLifecycleIntegrityError, match="disappeared"):
        lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A)


def test_harden_2b_idempotent_converged_outcome_after_winner_applied_candidate() -> None:
    """Reload + reapply original candidate yields unchanged when winner already merged."""
    tenant_id = "harden-2b-idempotent-converge"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )

    persistence = document_store_problem_persistence_for_tests(InMemoryDocumentStore())
    lifecycle = ProblemLifecycleEngine(persistence)
    try:
        persistence.create(baseline)
        grouping = _singleton_grouping_result(
            tenant_id=tenant_id,
            member=subject_a,
            signature=signature,
            observed_at=_OBSERVED_AT_A,
        )
        first = lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A)
        assert len(first.updated) == 1
        second = lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A)
        assert second.updated == ()
        assert len(second.unchanged) == 1
        assert second.unchanged[0].occurrence_count == 2
    finally:
        persistence.close()


def test_harden_2d_concurrent_identical_resolve_converges_once() -> None:
    tenant_id = "harden-2d-resolve-tenant"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )
    store = InMemoryDocumentStore()
    resolve_barrier = threading.Barrier(2)
    assert baseline.record_version == 1

    seed = document_store_problem_persistence_for_tests(store)
    seed.create(baseline)
    seed.close()

    resolved_results: list[Problem] = []
    errors: list[BaseException] = []

    def _resolve() -> None:
        persistence = _SynchronizedResolvePersistence(
            store,
            resolve_barrier=resolve_barrier,
            synchronized_expected_version=baseline.record_version,
        )
        lifecycle = ProblemLifecycleEngine(persistence)
        try:
            resolved_results.append(
                lifecycle.resolve(
                    tenant_id=tenant_id,
                    problem_id=baseline.problem_id,
                    resolved_at=_RESOLVED_AT,
                ),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_resolve), executor.submit(_resolve)]
        for future in futures:
            future.result(timeout=10)

    assert errors == []
    assert len(resolved_results) == 2
    assert all(result.status is ProblemStatus.RESOLVED for result in resolved_results)
    assert all(result.record_version == 2 for result in resolved_results)

    verifier = document_store_problem_persistence_for_tests(store)
    try:
        final = verifier.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final is not None
        assert final.status is ProblemStatus.RESOLVED
        assert final.record_version == 2
        assert final.occurrence_count == baseline.occurrence_count
    finally:
        verifier.close()


def test_harden_2d_resolve_retry_exhaustion_raises_lifecycle_integrity_error() -> None:
    store = _AlwaysConflictReplaceStore()
    tenant_id = "harden-2d-resolve-retry-exhaustion"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )

    persistence = document_store_problem_persistence_for_tests(store)
    lifecycle = ProblemLifecycleEngine(persistence)
    try:
        persistence.create(baseline)
        with pytest.raises(ProblemLifecycleIntegrityError) as exc_info:
            lifecycle.resolve(
                tenant_id=tenant_id,
                problem_id=baseline.problem_id,
                resolved_at=_RESOLVED_AT,
            )
        assert isinstance(exc_info.value.__cause__, ProblemPersistenceConflictError)
        final = persistence.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final == baseline
    finally:
        persistence.close()


def test_harden_2d_resolve_race_with_concurrent_occurrence_update_preserves_latest() -> None:
    tenant_id = "harden-2d-resolve-vs-update"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    baseline_subject = _sample_subject_ref(tenant_id=tenant_id)
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    baseline = sample_problem(
        tenant_id=tenant_id,
        subject_refs=(baseline_subject,),
        reconciliation_key=reconciliation_key,
    )
    store = InMemoryDocumentStore()
    update_barrier = threading.Barrier(2)
    assert baseline.record_version == 1

    seed = document_store_problem_persistence_for_tests(store)
    seed.create(baseline)
    seed.close()

    grouping = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject_a,
        signature=signature,
        observed_at=_OBSERVED_AT_A,
    )

    race_errors: list[BaseException] = []
    reconcile_result: list[ProblemLifecycleResult] = []

    def _resolve() -> None:
        persistence = _SynchronizedResolvePersistence(
            store,
            resolve_barrier=update_barrier,
            synchronized_expected_version=baseline.record_version,
        )
        lifecycle = ProblemLifecycleEngine(persistence)
        try:
            lifecycle.resolve(
                tenant_id=tenant_id,
                problem_id=baseline.problem_id,
                resolved_at=_RESOLVED_AT,
            )
        except BaseException as exc:  # noqa: BLE001
            race_errors.append(exc)
        finally:
            persistence.close()

    def _reconcile() -> None:
        persistence = _SynchronizedUpdatePersistence(
            store,
            update_barrier=update_barrier,
            synchronized_expected_version=baseline.record_version,
        )
        lifecycle = ProblemLifecycleEngine(persistence)
        try:
            reconcile_result.append(
                lifecycle.reconcile(grouping, observed_at=_OBSERVED_AT_A),
            )
        except BaseException as exc:  # noqa: BLE001
            race_errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_resolve),
            executor.submit(_reconcile),
        ]
        for future in futures:
            future.result(timeout=10)

    assert race_errors == []
    assert len(reconcile_result) == 1

    verifier = document_store_problem_persistence_for_tests(store)
    try:
        final = verifier.get(tenant_id=tenant_id, problem_id=baseline.problem_id)
        assert final is not None
        assert final.status is ProblemStatus.RESOLVED
        assert final.record_version == 3
        assert final.occurrence_count == 2
        assert subject_a in {occurrence.subject_ref for occurrence in final.occurrences}
    finally:
        verifier.close()


def test_harden_2d_create_race_converges_while_winner_canonical_pending() -> None:
    tenant_id = "harden-2d-create-race-lifecycle"
    signature = _sample_signature()
    reconciliation_key = _sample_reconciliation_key(
        tenant_id=tenant_id,
        signature=signature,
    )
    subject_a = _sample_subject_ref(tenant_id=tenant_id)
    subject_b = _sample_subject_ref(tenant_id=tenant_id)

    class _CreateRaceInterleavingStore(InMemoryDocumentStore):
        def __init__(self) -> None:
            super().__init__()
            self._entry_barrier = threading.Barrier(2)
            self._release_canonical = threading.Event()
            self._canonical_written = threading.Event()
            self._put_if_absent_calls = 0
            self._state_lock = threading.Lock()

        def release_canonical(self) -> None:
            self._release_canonical.set()

        def wait_canonical_written(self, *, timeout: float) -> bool:
            return self._canonical_written.wait(timeout=timeout)

        def put_if_absent(self, document: DocumentRecord) -> bool:
            with self._state_lock:
                self._put_if_absent_calls += 1
                call_number = self._put_if_absent_calls
            if call_number <= 2:
                self._entry_barrier.wait(timeout=5)
            if document.row_key.startswith("record:"):
                self._release_canonical.wait(timeout=5)
            inserted = super().put_if_absent(document)
            if document.row_key.startswith("record:") and inserted:
                self._canonical_written.set()
            return inserted

    store = _CreateRaceInterleavingStore()

    class _ReleaseOnPendingWinnerLookupPersistence(DocumentStoreProblemPersistence):
        def find_by_reconciliation_key(self, *, tenant_id: str, reconciliation_key):
            try:
                return super().find_by_reconciliation_key(
                    tenant_id=tenant_id,
                    reconciliation_key=reconciliation_key,
                )
            except ProblemPersistenceIntegrityError as exc:
                if (
                    exc.reason
                    is ProblemPersistenceIntegrityReason.RECONCILIATION_WINNER_CANONICAL_PENDING
                ):
                    store.release_canonical()
                    if not store.wait_canonical_written(timeout=5):
                        raise
                raise

    grouping_a = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject_a,
        signature=signature,
        observed_at=_OBSERVED_AT_A,
    )
    grouping_b = _singleton_grouping_result(
        tenant_id=tenant_id,
        member=subject_b,
        signature=signature,
        observed_at=_OBSERVED_AT_B,
    )

    results: list[ProblemLifecycleResult] = []
    errors: list[BaseException] = []

    def _reconcile(grouping_result: ProblemGroupingResult, observed_at: datetime) -> None:
        persistence = _ReleaseOnPendingWinnerLookupPersistence(
            store,
            list_cursor_secret=TEST_PROBLEM_LIST_CURSOR_SECRET,
            document_query_cursor_codec=store.query_cursor_codec,
        )
        lifecycle = ProblemLifecycleEngine(persistence)
        try:
            results.append(
                lifecycle.reconcile(grouping_result, observed_at=observed_at),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            persistence.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_reconcile, grouping_a, _OBSERVED_AT_A),
            executor.submit(_reconcile, grouping_b, _OBSERVED_AT_B),
        ]
        for future in futures:
            future.result(timeout=10)

    assert errors == []
    assert len(results) == 2

    verifier = document_store_problem_persistence_for_tests(store)
    try:
        listed = query_all_problems_for_tenant(verifier, tenant_id)
        assert len(listed) == 1
        final = listed[0]
        assert final.occurrence_count == 2
        assert final.record_version == 2
        assert verifier.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        ) == final
        final_subjects = {occurrence.subject_ref for occurrence in final.occurrences}
        assert final_subjects == {subject_a, subject_b}
    finally:
        verifier.close()
