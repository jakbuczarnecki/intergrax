# © Artur Czarnecki. All rights reserved.

"""Distributed-safe position allocation for DocumentBackedRuntimeEventStore."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import (
    DocumentQueryPageV1,
    DocumentRecord,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-distributed"
_SEQUENCE_ROW_KEY = "__run_sequence__"


class _GapInducingDocumentStore(InMemoryDocumentStore):
    """Blocks the first event put_if_absent so a concurrent writer can consume a gap."""

    def __init__(self) -> None:
        super().__init__()
        self._first_event_put_started = threading.Event()
        self._release_first_event_put = threading.Event()

    def put_if_absent(self, document: DocumentRecord) -> bool:
        if document.row_key != _SEQUENCE_ROW_KEY and not self._first_event_put_started.is_set():
            self._first_event_put_started.set()
            self._release_first_event_put.wait(timeout=5)
        return super().put_if_absent(document)

    def release_first_event_put(self) -> None:
        self._release_first_event_put.set()


def _event(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    execution_id: str | None = None,
    event_id: str | None = None,
) -> RuntimeEvent:
    identity = runtime_event_test_identity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=_TENANT,
        **identity,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        correlation_id=run_id,
    )


def _paired_stores() -> tuple[DocumentBackedRuntimeEventStore, DocumentBackedRuntimeEventStore]:
    backend = InMemoryDocumentStore()
    return (
        DocumentBackedRuntimeEventStore(backend),
        DocumentBackedRuntimeEventStore(backend),
    )


def test_requires_conditional_document_store() -> None:
    class _PlainDocumentStore:
        def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
            return None

        def put(self, document: DocumentRecord) -> None:
            pass

        def delete(self, partition_key: str, row_key: str) -> None:
            pass

        def query(
            self,
            partition_key: str,
            *,
            limit: int = 100,
            row_key_prefix: str | None = None,
            cursor: str | None = None,
        ) -> DocumentQueryPageV1:
            return DocumentQueryPageV1()

        def close(self) -> None:
            pass

    plain = _PlainDocumentStore()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentBackedRuntimeEventStore(plain)


def test_two_store_instances_allocate_distinct_positions_for_different_events() -> None:
    store_a, store_b = _paired_stores()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_a = _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    event_b = _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    barrier = threading.Barrier(2)
    results: list[int] = []
    lock = threading.Lock()

    def _append_a() -> None:
        barrier.wait()
        position = store_a.append(event_a, tenant_id=_TENANT).position.value
        with lock:
            results.append(position)

    def _append_b() -> None:
        barrier.wait()
        position = store_b.append(event_b, tenant_id=_TENANT).position.value
        with lock:
            results.append(position)

    threads = [
        threading.Thread(target=_append_a),
        threading.Thread(target=_append_b),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert len(set(results)) == 2
    assert set(results) == {1, 2}
    ordered = store_a.list_positioned_for_run(run_id, tenant_id=_TENANT)
    assert {row.event_id for row in ordered} == {event_a.event_id, event_b.event_id}


def test_concurrent_same_event_id_returns_single_accepted_position() -> None:
    store_a, store_b = _paired_stores()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_id = mint_event_id()
    event = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_id=event_id,
    )
    barrier = threading.Barrier(2)
    results: list[int] = []
    lock = threading.Lock()

    def _append(store: DocumentBackedRuntimeEventStore) -> None:
        barrier.wait()
        position = store.append(event, tenant_id=_TENANT).position.value
        with lock:
            results.append(position)

    threads = [
        threading.Thread(target=_append, args=(store_a,)),
        threading.Thread(target=_append, args=(store_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] == results[1]
    assert len(store_a.list_positioned_for_run(run_id, tenant_id=_TENANT)) == 1


def test_multi_writer_stress_unique_strictly_increasing_positions() -> None:
    backend = InMemoryDocumentStore()
    stores = [
        DocumentBackedRuntimeEventStore(backend),
        DocumentBackedRuntimeEventStore(backend),
        DocumentBackedRuntimeEventStore(backend),
    ]
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_count = 30
    events = [
        _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
        for _ in range(event_count)
    ]
    barrier = threading.Barrier(len(events))
    results: list[int] = []
    lock = threading.Lock()

    def _append(index: int) -> None:
        store = stores[index % len(stores)]
        barrier.wait()
        position = store.append(events[index], tenant_id=_TENANT).position.value
        with lock:
            results.append(position)

    with ThreadPoolExecutor(max_workers=event_count) as pool:
        list(pool.map(_append, range(event_count)))

    assert len(results) == event_count
    assert len(set(results)) == event_count
    assert min(results) >= 1
    ordered = stores[0].list_positioned_for_run(
        run_id,
        tenant_id=_TENANT,
        limit=1000,
    )
    assert len(ordered) == event_count
    ordered_positions = [row.position.value for row in ordered]
    assert ordered_positions == sorted(ordered_positions)
    assert len(set(ordered_positions)) == event_count


def test_concurrent_same_event_id_may_leave_unused_position_gap() -> None:
    backend = _GapInducingDocumentStore()
    store_a = DocumentBackedRuntimeEventStore(backend)
    store_b = DocumentBackedRuntimeEventStore(backend)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    event_id = mint_event_id()
    event = _event(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_id=event_id,
    )
    follow_up = _event(task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    accepted_positions: list[int] = []
    lock = threading.Lock()
    append_error: list[BaseException] = []

    def _append_a() -> None:
        try:
            position = store_a.append(event, tenant_id=_TENANT).position.value
            with lock:
                accepted_positions.append(position)
        except BaseException as exc:
            append_error.append(exc)

    def _append_b() -> None:
        try:
            position = store_b.append(event, tenant_id=_TENANT).position.value
            with lock:
                accepted_positions.append(position)
        except BaseException as exc:
            append_error.append(exc)

    thread_a = threading.Thread(target=_append_a)
    thread_b = threading.Thread(target=_append_b)
    thread_a.start()
    backend._first_event_put_started.wait(timeout=5)
    thread_b.start()
    backend.release_first_event_put()
    thread_a.join()
    thread_b.join()

    assert append_error == []
    assert len(accepted_positions) == 2
    assert accepted_positions[0] == accepted_positions[1]
    accepted_position = accepted_positions[0]
    assert accepted_position >= 2
    next_position = store_a.append(follow_up, tenant_id=_TENANT).position.value
    assert next_position > accepted_position
    ordered = store_a.list_positioned_for_run(run_id, tenant_id=_TENANT, limit=10)
    ordered_positions = sorted(row.position.value for row in ordered)
    assert ordered_positions == sorted({accepted_position, next_position})
    assert len(ordered_positions) == 2
    assert ordered_positions[0] != 1
