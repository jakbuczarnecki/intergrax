# © Artur Czarnecki. All rights reserved.

"""F-D7-08 — retry back-off rows must not hide fresh work."""

from __future__ import annotations

import time

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store import document_store_task_queue as dstq
from tests.qualification.pdc_x.d7_repair._queue_support import TASK, TENANT, backoff_policy, request

pytestmark = [pytest.mark.unit]


def _back_off(queue: DocumentStoreTaskQueue, count: int) -> list:
    handles = [queue.enqueue(request()) for _ in range(count)]
    claimed = queue.claim_pending(tenant_id=TENANT, limit=count)
    assert len(claimed) == count
    for handle in handles:
        status = queue.retry_or_dead_letter(handle, error_message="flaky", retry_policy=backoff_policy())
        assert status is TaskStatus.PENDING  # back-off one hour, not elapsed
    return handles


@pytest.mark.parametrize("count", [40, 200])
def test_fresh_task_is_claimed_behind_any_number_of_backed_off_rows(count: int) -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    backed_off = _back_off(queue, count)
    fresh = queue.enqueue(request())

    claimed = queue.claim_pending(tenant_id=TENANT, limit=4)

    assert [handle.task_id for handle, _ in claimed] == [fresh.task_id]
    for handle in backed_off:
        assert queue.get_status(handle) is TaskStatus.PENDING  # still waiting, not claimed early


def test_backed_off_rows_become_claimable_once_the_back_off_has_elapsed(monkeypatch) -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    backed_off = _back_off(queue, 5)
    assert queue.claim_pending(tenant_id=TENANT, limit=10) == []
    real_time = time.time
    monkeypatch.setattr(dstq.time, "time", lambda: real_time() + 7200.0)  # two hours later
    claimed = queue.claim_pending(tenant_id=TENANT, limit=10)
    assert sorted(handle.task_id for handle, _ in claimed) == sorted(h.task_id for h in backed_off)


def test_tenant_filter_still_applies_behind_backed_off_rows() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    _back_off(queue, 40)
    other = queue.enqueue(request(tenant="t-other"))
    mine = queue.enqueue(request())
    claimed = queue.claim_pending(tenant_id="t-other", limit=10)
    assert [handle.task_id for handle, _ in claimed] == [other.task_id]
    assert [handle.task_id for handle, _ in queue.claim_pending(tenant_id=TENANT, limit=10)] == [mine.task_id]


def test_legacy_back_off_rows_in_the_pending_index_do_not_hide_fresh_work() -> None:
    """Rows written before the repair sit in the pending index; they must drain, not block."""
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    future = time.time() + 3600.0
    for n in range(120):
        task_id = f"dstq_!legacy{n:04d}"  # sorts before every fresh id
        store.put(
            DocumentRecord(
                partition_key=dstq._partition(TENANT),
                row_key=task_id,
                data={
                    "task_id": task_id,
                    "tenant_id": TENANT,
                    "run_id": "r",
                    "task_name": TASK,
                    "payload_base64": "e30=",
                    "idempotency_key": None,
                    "status": TaskStatus.PENDING.value,
                    "attempts": 1,
                    "created_at_epoch": 1.0,
                    "updated_at_epoch": 1.0,
                    "available_at_epoch": future,
                    "error_message": "flaky",
                    "output_base64": None,
                    "identity": None,
                    "dead_letter": None,
                },
            )
        )
        store.put(
            DocumentRecord(
                partition_key=dstq._PENDING_INDEX_PARTITION,
                row_key=dstq._index_row(TENANT, task_id),
                data={"tenant_id": TENANT, "task_id": task_id},
            )
        )
    fresh = queue.enqueue(request())
    claimed = queue.claim_pending(tenant_id=TENANT, limit=4)
    assert [handle.task_id for handle, _ in claimed] == [fresh.task_id]


def test_claim_work_is_bounded_by_the_requested_window_not_the_backlog() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    _back_off(queue, 300)
    queue.enqueue(request())
    queue.claim_pending(tenant_id=TENANT, limit=4)
    # The claim path must not examine the whole backed-off backlog in one call.
    assert store.last_query_rows_examined < 300
