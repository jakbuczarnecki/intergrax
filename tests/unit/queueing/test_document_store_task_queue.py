# © Artur Czarnecki. All rights reserved.

"""Unit tests for DocumentStore-backed durable TaskQueue."""

from __future__ import annotations

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel


class _Out(BaseModel):
    ok: bool = True


def test_enqueue_persists_pending_and_claim_runs() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="t1",
            run_id="r1",
            task_name="demo.task",
            payload=b'{"x":1}',
            idempotency_key="idem-1",
        )
    )
    assert queue.get_status(handle) is TaskStatus.PENDING
    claimed = queue.claim_pending(limit=1)
    assert len(claimed) == 1
    assert queue.get_status(handle) is TaskStatus.RUNNING
    queue.mark_succeeded(handle, output=b"{}")
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED


def test_idempotent_enqueue_reuses_task_id() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    first = queue.enqueue(
        TaskRequest(
            tenant_id="t1",
            run_id="r1",
            task_name="demo.task",
            payload=b"{}",
            idempotency_key="same",
        )
    )
    second = queue.enqueue(
        TaskRequest(
            tenant_id="t1",
            run_id="r2",
            task_name="demo.task",
            payload=b"{}",
            idempotency_key="same",
        )
    )
    assert first.task_id == second.task_id


def test_recover_interrupted_running_fail_closed() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="t1",
            run_id="r1",
            task_name="demo.task",
            payload=b"{}",
        )
    )
    queue.claim_pending(limit=1)
    interrupted = queue.recover_interrupted_running()
    assert len(interrupted) == 1
    assert queue.get_status(handle) is TaskStatus.FAILED


def test_worker_executes_registered_handler() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key=None,
        execution_identity: BackgroundExecutionIdentity,
    ):
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.ok(_Out())

    registry.register("demo.task", handler)
    queue.enqueue(
        TaskRequest(
            tenant_id="t1",
            run_id="r1",
            task_name="demo.task",
            payload=b"{}",
        )
    )
    worker = DocumentStoreTaskWorker(queue, registry)
    assert worker.drain_once() == 1
    rows = queue.list_tasks("t1")
    assert rows
    assert rows[0].status is TaskStatus.SUCCEEDED
