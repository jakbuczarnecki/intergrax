# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-E NT-RW-REC-01 (EX-L-06) and NT-RW-REC-02 (EX-L-03): no second consumer, no lost terminal row.

REC-01: a ``stop()`` that times out while a handler is still running must not let a later ``start()``
run a second consumer next to the old one.

REC-02: ``recover_interrupted_running`` is a compare-and-set per row: a task that a concurrent writer
finished between the recovery read and the recovery write keeps its terminal status.
"""

from __future__ import annotations

import threading

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel

pytestmark = [pytest.mark.unit]

_TENANT = "t-rw-rec"
_TASK_NAME = "rw.rec.blocking"
_WORKER_THREAD_NAME = "intergrax-document-store-task-worker"
# The running index partition of the document-store queue (``_RUNNING_INDEX_PARTITION``).
_RUNNING_INDEX = "intergrax.task_queue.v1:__running_index__"
_WAIT = 10.0


class _Out(BaseModel):
    ok: bool = True


class _BlockingHandler:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BaseModel]:
        del tenant_id, run_id, payload, idempotency_key, execution_identity
        with self._lock:
            self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=_WAIT), "handler was never released"
        return ToolExecutionResult.ok(_Out())


def _worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _WORKER_THREAD_NAME and t.is_alive()]


def _request() -> TaskRequest:
    return TaskRequest(tenant_id=_TENANT, run_id="run-1", task_name=_TASK_NAME, payload=b"{}")


def _worker(
    store: InMemoryDocumentStore, handler: _BlockingHandler
) -> tuple[DocumentStoreTaskWorker, DocumentStoreTaskQueue]:
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    registry.register(_TASK_NAME, handler)
    admission = wire_background_execution_admission_dependencies(document_store=store)
    worker = DocumentStoreTaskWorker(
        queue,
        registry,
        poll_interval_seconds=0.01,
        identity_persistence=admission.identity_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        causal_evidence_persistence=wire_causal_evidence_persistence(document_store=store),
    )
    return worker, queue


def test_rw_rec_01_stop_then_start_never_two_consumers() -> None:
    assert _worker_threads() == []
    store = InMemoryDocumentStore()
    handler = _BlockingHandler()
    worker, queue = _worker(store, handler)
    handle = queue.enqueue(_request())
    worker.start()
    try:
        assert handler.entered.wait(timeout=_WAIT)

        stopped = worker.stop(timeout_seconds=0.05)

        assert stopped is False, "a stop() that timed out must report that the consumer still runs"
        from intergrax.queueing.providers.document_store.colocated_worker import (
            DocumentStoreTaskWorkerStillRunning,
        )

        with pytest.raises(DocumentStoreTaskWorkerStillRunning):
            worker.start()
        assert len(_worker_threads()) == 1, "start() after an unfinished stop() ran a second consumer"
        assert handler.calls == 1
    finally:
        handler.release.set()

    assert worker.stop(timeout_seconds=_WAIT) is True
    assert _worker_threads() == []
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED

    worker.start()  # the old consumer has ended: a new start works again
    try:
        assert len(_worker_threads()) == 1
    finally:
        assert worker.stop(timeout_seconds=_WAIT) is True
    assert handler.calls == 1


class _ConcurrentFinishStore(InMemoryDocumentStore):
    """Completes one task through the queue right after the recovery read of its row."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: DocumentStoreTaskQueue | None = None
        self.armed: TaskHandle | None = None

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        record = super().get(partition_key, row_key)
        armed = self.armed
        if armed is not None and row_key == armed.task_id and record is not None and self.queue is not None:
            self.armed = None
            self.queue.mark_succeeded(armed, output=None)  # the concurrent writer wins the race
        return record


def test_rw_rec_02_recovery_is_compare_and_set() -> None:
    store = _ConcurrentFinishStore()
    queue = DocumentStoreTaskQueue(store)
    store.queue = queue
    handle = queue.enqueue(_request())
    claimed = queue.claim_pending(limit=1)
    assert [h.task_id for h, _ in claimed] == [handle.task_id]
    assert queue.get_status(handle) is TaskStatus.RUNNING

    store.armed = handle
    interrupted = queue.recover_interrupted_running()

    assert store.armed is None, "the concurrent completion was not triggered"
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED, "recovery overwrote a terminal row"
    assert interrupted == []
    assert list(store.query(_RUNNING_INDEX, limit=10).documents) == []
    assert queue.recover_interrupted_running() == []


class _SettleFailsQueue(DocumentStoreTaskQueue):
    """Queue whose ``mark_failed`` raises, so settling a failed row escapes ``drain_once``."""

    def mark_failed(self, handle: TaskHandle, *, error_message: str) -> None:
        del handle, error_message
        raise RuntimeError("store unavailable while settling")


def test_rw_rec_01b_in_flight_count_survives_a_failing_settlement() -> None:
    store = InMemoryDocumentStore()
    queue = _SettleFailsQueue(store)
    registry = TaskExecutionRegistry()

    def _failing_handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None = None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BaseModel]:
        del tenant_id, run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.fail("rw_rec_failed", "handler failed")

    registry.register(_TASK_NAME, _failing_handler)
    admission = wire_background_execution_admission_dependencies(document_store=store)
    worker = DocumentStoreTaskWorker(
        queue,
        registry,
        identity_persistence=admission.identity_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        causal_evidence_persistence=wire_causal_evidence_persistence(document_store=store),
    )
    queue.enqueue(_request())
    queue.enqueue(TaskRequest(tenant_id=_TENANT, run_id="run-2", task_name=_TASK_NAME, payload=b"{}"))

    with pytest.raises(RuntimeError, match="store unavailable while settling"):
        worker.drain_once()

    assert worker.in_flight_count() == 0, "an escaped settlement left rows counted as in flight"
    assert worker.wait_idle(0.01) is True
