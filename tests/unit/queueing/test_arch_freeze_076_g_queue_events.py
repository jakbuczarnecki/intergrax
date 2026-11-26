# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package G — durable queue claim and terminal events on RuntimeEventBus."""

from __future__ import annotations

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskExecutionIdentity, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store.document_store_task_queue import DocumentStoreTaskQueue
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit]


def _canonical_identity() -> TaskExecutionIdentity:
    ids = runtime_event_test_identity()
    return TaskExecutionIdentity(
        task_id=str(ids["task_id"]),
        run_id=str(ids["run_id"]),
        attempt_id=str(ids["attempt_id"]),
        execution_id=str(ids["execution_id"]),
    )


def test_claim_and_success_emit_runtime_events_with_row_identity() -> None:
    store = InMemoryDocumentStore()
    bus = RuntimeEventBus(record_history=True)
    queue = DocumentStoreTaskQueue(store, runtime_event_bus=bus)
    tenant_id = "tenant-af-g"

    handle = queue.enqueue(
        TaskRequest(
            tenant_id=tenant_id,
            run_id="transport-run-1",
            task_name="arch_freeze.queue.demo",
            payload=b"{}",
            identity=_canonical_identity(),
        )
    )
    document_task_id = handle.task_id

    claimed = queue.claim_pending(tenant_id=tenant_id, limit=1)
    assert len(claimed) == 1
    assert queue.get_status(handle) is TaskStatus.RUNNING

    queue.mark_succeeded(handle, output=b"{}")
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED

    events = bus.history
    assert len(events) == 2
    progress, completed = events
    assert progress.event_type is RuntimeEventType.TASK_PROGRESS
    assert completed.event_type is RuntimeEventType.TASK_COMPLETED
    assert progress.tenant_id == tenant_id
    assert completed.tenant_id == tenant_id
    assert progress.payload["document_task_id"] == document_task_id
    assert completed.payload["document_task_id"] == document_task_id
    assert progress.payload["tenant_id"] == tenant_id
    assert completed.payload["tenant_id"] == tenant_id


def test_queue_without_bus_claims_and_settles_silently() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    tenant_id = "tenant-silent"

    handle = queue.enqueue(
        TaskRequest(
            tenant_id=tenant_id,
            run_id="transport-run-2",
            task_name="arch_freeze.queue.demo",
            payload=b"{}",
            identity=_canonical_identity(),
        )
    )
    claimed = queue.claim_pending(tenant_id=tenant_id, limit=1)
    assert len(claimed) == 1
    queue.mark_succeeded(handle, output=b"{}")
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED
