# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-READ-R1-R3 active-vs-abandoned append-intent safety proofs."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_append_intent import (
    FunctionalEvidenceAppendFaultBoundary,
    FunctionalEvidenceAppendIntentStore,
    functional_evidence_append_pending_row_key,
)
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    encode_execution_index_v2,
    execution_index_v2_row_key_from_evidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceProjectionConsistencyPendingError,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_evidence_projection_state import (
    FunctionalEvidenceProjectionState,
    encode_functional_evidence_projection_state,
    functional_evidence_projection_state_row_key,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    encode_functional_evidence_record,
)

pytestmark = pytest.mark.unit

_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 9, 4, 16, 0, tzinfo=UTC)
_PARTITION_PREFIX = "intergrax.functional_evidence.v1"


class _SingleShotAppendFaultInjector:
    def __init__(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> None:
        self._boundary = boundary
        self._fired = False

    def should_fault_after(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> bool:
        if not self._fired and boundary is self._boundary:
            self._fired = True
            return True
        return False


class _ActiveWriterSyncInjector:
    """Blocks after intent until released; optionally faults after canonical."""

    def __init__(self) -> None:
        self.intent_created = threading.Event()
        self.allow_canonical = threading.Event()
        self._passed_intent = False
        self._fault_after_canonical = False

    def enable_fault_after_canonical(self) -> None:
        self._fault_after_canonical = True

    def should_fault_after(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> bool:
        if boundary is FunctionalEvidenceAppendFaultBoundary.AFTER_INTENT and not self._passed_intent:
            self.intent_created.set()
            if not self.allow_canonical.wait(timeout=5):
                raise TimeoutError("active writer sync timed out waiting for allow_canonical")
            self._passed_intent = True
            return False
        if (
            boundary is FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL
            and self._fault_after_canonical
        ):
            return True
        return False


class _CountingDocumentStore(InMemoryDocumentStore):
    def __init__(self) -> None:
        super().__init__(cursor_secret=_CURSOR_SECRET)
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


def _partition_key(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _persistence(
    store: InMemoryDocumentStore,
    *,
    fault_injector: _SingleShotAppendFaultInjector | _ActiveWriterSyncInjector | None = None,
) -> DocumentStoreFunctionalEvidencePersistence:
    return DocumentStoreFunctionalEvidencePersistence(
        store,
        cursor_secret=_CURSOR_SECRET,
        append_fault_injector=fault_injector,
    )


def _seed_healthy_execution(
    persistence: DocumentStoreFunctionalEvidencePersistence,
    scope,
    count: int,
) -> tuple:
    fixtures = tuple(
        sample_functional_evidence(
            scope=scope,
            operation_name=f"healthy-{index}",
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        )
        for index in range(count)
    )
    for evidence in fixtures:
        persistence.append(evidence)
    return fixtures


def test_reader_sees_active_pending_before_canonical_fails_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-active-pending")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    sync = _ActiveWriterSyncInjector()
    writer_error: list[BaseException] = []

    def _writer() -> None:
        try:
            _persistence(store, fault_injector=sync).append(evidence)
        except BaseException as exc:  # noqa: BLE001
            writer_error.append(exc)

    writer_thread = threading.Thread(target=_writer)
    writer_thread.start()
    assert sync.intent_created.wait(timeout=5)
    reader = _persistence(store)
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError, match="unresolved"):
        collect_all_evidence(
            reader,
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    assert FunctionalEvidenceAppendIntentStore(store).has_pending_for_execution(
        partition_key=_partition_key(scope.tenant_id),
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    sync.enable_fault_after_canonical()
    sync.allow_canonical.set()
    writer_thread.join(timeout=10)
    assert writer_error and isinstance(writer_error[0], FunctionalEvidencePersistenceIntegrityError)


def test_pending_remains_after_reader_fail_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-pending-durable")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_INTENT)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        _persistence(store, fault_injector=fault).append(evidence)
    intent_store = FunctionalEvidenceAppendIntentStore(store)
    assert intent_store.has_pending_for_execution(
        partition_key=_partition_key(scope.tenant_id),
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    assert intent_store.has_pending_for_execution(
        partition_key=_partition_key(scope.tenant_id),
        task_id=scope.task_id,
        run_id=scope.run_id,
    )


def test_writer_resumes_after_reader_fail_closed() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-writer-resume")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_INTENT)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        _persistence(store, fault_injector=fault).append(evidence)
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    assert _persistence(store).append(evidence) == evidence
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_writer_crashes_after_canonical_then_fresh_reader_repairs() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-crash-after-canonical")
    base = _seed_healthy_execution(_persistence(store), scope, 3)
    evidence = sample_functional_evidence(
        scope=scope,
        operation_name="late",
        recorded_at=_BASE_TIME + timedelta(seconds=100),
    )
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        _persistence(store, fault_injector=fault).append(evidence)
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    expected = tuple(sorted((*base, evidence), key=functional_evidence_query_order_key))
    assert collected == expected


def test_abandoned_pending_blocks_query() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-abandoned")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
    FunctionalEvidenceAppendIntentStore(store).create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(evidence.evidence_id),
    )
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )


def test_retry_abandoned_intent_completes_successfully() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-abandoned-retry")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
    FunctionalEvidenceAppendIntentStore(store).create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(evidence.evidence_id),
    )
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    assert _persistence(store).append(evidence) == evidence
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_concurrent_retry_same_evidence() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-concurrent-same")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        _persistence(store, fault_injector=fault).append(evidence)
    barrier = threading.Barrier(2)
    results: list = []
    errors: list[BaseException] = []

    def _retry() -> None:
        persistence = _persistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(evidence))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_retry), executor.submit(_retry)]
        for future in futures:
            future.result(timeout=30)
    assert not errors
    assert len(results) == 2
    assert results[0] == evidence
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_concurrent_different_evidence_same_execution() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-concurrent-diff")
    evidence_a = sample_functional_evidence(
        scope=scope,
        operation_name="a",
        recorded_at=_BASE_TIME,
    )
    evidence_b = sample_functional_evidence(
        scope=scope,
        operation_name="b",
        recorded_at=_BASE_TIME + timedelta(seconds=1),
    )
    barrier = threading.Barrier(2)
    results: list = []
    errors: list[BaseException] = []

    def _append(item) -> None:
        persistence = _persistence(store)
        try:
            barrier.wait(timeout=5)
            results.append(persistence.append(item))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_append, evidence_a),
            executor.submit(_append, evidence_b),
        ]
        for future in futures:
            future.result(timeout=30)
    assert not errors
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert {item.evidence_id for item in collected} == {
        evidence_a.evidence_id,
        evidence_b.evidence_id,
    }


def test_corrupt_intent_fails_integrity_not_consistency_pending() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-corrupt-intent")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=functional_evidence_append_pending_row_key(
                task_id=scope.task_id,
                run_id=scope.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data={"schema_version": "broken", "evidence_id": str(evidence.evidence_id)},
        ),
    )
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError, match="append intent"):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )


def test_pending_plus_canonical_repairs() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-pending-canonical")
    base = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    pending = sample_functional_evidence(
        scope=scope,
        operation_name="pending",
        recorded_at=_BASE_TIME + timedelta(seconds=1),
    )
    persistence = _persistence(store)
    persistence.append(base)
    partition_key = _partition_key(scope.tenant_id)
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{pending.evidence_id}",
            data=encode_functional_evidence_record(pending),
        ),
    )
    FunctionalEvidenceAppendIntentStore(store).create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(pending.evidence_id),
    )
    collected = collect_all_evidence(
        persistence,
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    expected = tuple(sorted((base, pending), key=functional_evidence_query_order_key))
    assert collected == expected


def test_pending_plus_canonical_plus_v2_repairs_v1() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-pending-v2")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{evidence.evidence_id}",
            data=encode_functional_evidence_record(evidence),
        ),
    )
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=encode_execution_index_v2(evidence),
        ),
    )
    FunctionalEvidenceAppendIntentStore(store).create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(evidence.evidence_id),
    )
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_healthy_fast_path_operation_count() -> None:
    store = _CountingDocumentStore()
    scope = sample_functional_evidence_scope(tenant_id="r1r3-fastpath")
    persistence = _persistence(store)
    _seed_healthy_execution(persistence, scope, 1000)
    store.query_calls = 0
    store.get_calls = 0
    persistence.query_evidence(
        FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            page_size=25,
        ),
    )
    assert store.get_calls <= 50
    assert store.get_calls < 200
    assert store.query_calls <= 5


def test_cursor_union_after_repair() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-cursor-union")
    persistence = _persistence(store)
    fixtures = _seed_healthy_execution(persistence, scope, 20)
    late = sample_functional_evidence(
        scope=scope,
        operation_name="late-crash",
        recorded_at=_BASE_TIME + timedelta(seconds=100),
    )
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL)
    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        _persistence(store, fault_injector=fault).append(late)
    expected_ids = {item.evidence_id for item in fixtures} | {late.evidence_id}
    collected: list = []
    cursor: str | None = None
    reader = _persistence(store)
    while True:
        page = reader.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                page_size=5,
                cursor=cursor,
            ),
        )
        collected.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert {item.evidence_id for item in collected} == expected_ids


def test_consistency_pending_is_not_integrity_error() -> None:
    assert issubclass(
        FunctionalEvidenceProjectionConsistencyPendingError,
        FunctionalEvidencePersistenceError,
    )
    assert not issubclass(
        FunctionalEvidenceProjectionConsistencyPendingError,
        FunctionalEvidencePersistenceIntegrityError,
    )


def test_conflict_same_evidence_id_different_payload() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-conflict")
    first = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    second = sample_functional_evidence(
        scope=scope,
        recorded_at=_BASE_TIME,
        evidence_id=str(first.evidence_id),
        operation_name="different-op",
    )
    persistence = _persistence(store)
    persistence.append(first)
    with pytest.raises(FunctionalEvidencePersistenceConflictError):
        persistence.append(second)


def test_manifest_complete_with_pending_intent_blocks_fast_path_then_repairs() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-manifest-pending")
    evidence = sample_functional_evidence(scope=scope, recorded_at=_BASE_TIME)
    partition_key = _partition_key(scope.tenant_id)
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=functional_evidence_projection_state_row_key(
                task_id=scope.task_id,
                run_id=scope.run_id,
            ),
            data=encode_functional_evidence_projection_state(
                FunctionalEvidenceProjectionState(
                    schema_version="intergrax.functional_evidence.projection_state.v1",
                    state="complete",
                    generation=1,
                    v1_rows_reconciled=0,
                ),
            ),
        ),
    )
    store.put(
        DocumentRecord(
            partition_key=partition_key,
            row_key=f"record:{evidence.evidence_id}",
            data=encode_functional_evidence_record(evidence),
        ),
    )
    FunctionalEvidenceAppendIntentStore(store).create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(evidence.evidence_id),
    )
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    assert collected == (evidence,)


def test_active_writer_reader_race_then_crash_after_canonical_recovery() -> None:
    store = InMemoryDocumentStore(cursor_secret=_CURSOR_SECRET)
    scope = sample_functional_evidence_scope(tenant_id="r1r3-race-recovery")
    base = _seed_healthy_execution(_persistence(store), scope, 5)
    evidence = sample_functional_evidence(
        scope=scope,
        operation_name="race",
        recorded_at=_BASE_TIME + timedelta(seconds=100),
    )
    sync = _ActiveWriterSyncInjector()
    writer_error: list[BaseException] = []

    def _writer() -> None:
        try:
            _persistence(store, fault_injector=sync).append(evidence)
        except BaseException as exc:  # noqa: BLE001
            writer_error.append(exc)

    writer_thread = threading.Thread(target=_writer)
    writer_thread.start()
    assert sync.intent_created.wait(timeout=5)
    with pytest.raises(FunctionalEvidenceProjectionConsistencyPendingError):
        collect_all_evidence(
            _persistence(store),
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
    sync.enable_fault_after_canonical()
    sync.allow_canonical.set()
    writer_thread.join(timeout=10)
    assert writer_error and isinstance(writer_error[0], FunctionalEvidencePersistenceIntegrityError)
    collected = collect_all_evidence(
        _persistence(store),
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
    )
    expected = tuple(sorted((*base, evidence), key=functional_evidence_query_order_key))
    assert collected == expected
