# © Artur Czarnecki. All rights reserved.

"""F-D7-07 — two queue instances over one conditional store must not double-claim or double-enqueue."""

from __future__ import annotations

import threading

import pytest
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store import document_store_task_queue as dstq
from tests.qualification.pdc_x.d7_repair._queue_support import (
    TENANT,
    PlainStore,
    SlowReadStore,
    request,
)

pytestmark = [pytest.mark.unit]


def _run_threads(count: int, target) -> None:
    barrier = threading.Barrier(count)

    def runner(index: int) -> None:
        barrier.wait()
        target(index)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()


def test_two_instances_never_claim_the_same_task() -> None:
    store = SlowReadStore()
    producer = DocumentStoreTaskQueue(store)
    handles = [producer.enqueue(request()) for _ in range(40)]
    instances = [DocumentStoreTaskQueue(store), DocumentStoreTaskQueue(store)]
    claimed: list[list[str]] = [[], []]

    def drain(index: int) -> None:
        while True:
            got = instances[index].claim_pending(tenant_id=TENANT, limit=4)
            if not got:
                return
            claimed[index].extend(handle.task_id for handle, _ in got)

    _run_threads(2, drain)

    everything = claimed[0] + claimed[1]
    assert len(everything) == len(set(everything)), "a task was claimed by both instances"
    assert sorted(everything) == sorted(h.task_id for h in handles)
    for handle in handles:
        assert producer.get_status(handle) is TaskStatus.RUNNING


def test_concurrent_idempotent_enqueue_over_two_instances_creates_one_task() -> None:
    store = SlowReadStore()
    instances = [DocumentStoreTaskQueue(store), DocumentStoreTaskQueue(store)]
    seen: list[TaskHandle] = []
    lock = threading.Lock()

    def enqueue(index: int) -> None:
        for _ in range(10):
            handle = instances[index].enqueue(request(key="same-key"))
            with lock:
                seen.append(handle)

    _run_threads(2, enqueue)

    assert len({handle.task_id for handle in seen}) == 1  # one task, whatever the interleaving
    assert len(instances[0].list_tasks(TENANT)) == 1


def test_late_terminal_write_from_a_second_instance_loses_the_compare_and_set() -> None:
    store = SlowReadStore()
    first, second = DocumentStoreTaskQueue(store), DocumentStoreTaskQueue(store)
    handle = first.enqueue(request())
    first.claim_pending(tenant_id=TENANT, limit=1)
    first.mark_succeeded(handle, output=b"ok")
    second.mark_failed(handle, error_message="stale instance")
    assert first.get_status(handle) is TaskStatus.SUCCEEDED


def test_orphan_idempotency_reservation_is_repaired_not_returned_as_a_live_handle() -> None:
    store = SlowReadStore()
    queue = DocumentStoreTaskQueue(store)
    # A crash between the reservation and the task write leaves a reservation without a task row.
    store.put(
        DocumentRecord(
            partition_key=dstq._partition(TENANT),
            row_key=f"{dstq._IDEMPOTENCY_PREFIX}orphan-key",
            data={"task_id": "dstq_never_written"},
        )
    )
    handle = queue.enqueue(request(key="orphan-key"))
    assert handle.task_id != "dstq_never_written"
    assert queue.get_status_if_present(handle) is TaskStatus.PENDING  # the task row really exists
    again = queue.enqueue(request(key="orphan-key"))
    assert again.task_id == handle.task_id  # and the key now resolves to it


def test_plain_document_store_keeps_the_single_instance_behaviour() -> None:
    queue = DocumentStoreTaskQueue(PlainStore())  # type: ignore[arg-type]
    first = queue.enqueue(request(key="k"))
    assert queue.enqueue(request(key="k")).task_id == first.task_id
    claimed = queue.claim_pending(tenant_id=TENANT, limit=4)
    assert [handle.task_id for handle, _ in claimed] == [first.task_id]
    queue.mark_succeeded(first, output=b"x")
    assert queue.get_status(first) is TaskStatus.SUCCEEDED
