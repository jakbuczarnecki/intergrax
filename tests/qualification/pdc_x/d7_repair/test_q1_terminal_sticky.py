# © Artur Czarnecki. All rights reserved.

"""F-D7-06 — terminal queue states are sticky: no later transition may leave them."""

from __future__ import annotations

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import DeadLetterReason, TaskHandle, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from tests.qualification.pdc_x.d7_repair._queue_support import backoff_policy, claim_one, request

pytestmark = [pytest.mark.unit]


def _running(queue: DocumentStoreTaskQueue) -> TaskHandle:
    handle = queue.enqueue(request())
    claim_one(queue, handle)
    assert queue.get_status(handle) is TaskStatus.RUNNING
    return handle


def _to_succeeded(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> None:
    queue.mark_succeeded(handle, output=b"done")


def _to_failed(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> None:
    queue.mark_failed(handle, error_message="boom")


def _to_dead_lettered(queue: DocumentStoreTaskQueue, handle: TaskHandle) -> None:
    queue.mark_dead_lettered(handle, reason=DeadLetterReason.RETRY_BUDGET_EXHAUSTED, error_message="poison")


_TERMINALS = {
    TaskStatus.SUCCEEDED: _to_succeeded,
    TaskStatus.FAILED: _to_failed,
    TaskStatus.DEAD_LETTERED: _to_dead_lettered,
}


def _late_succeeded(queue, handle) -> None:
    queue.mark_succeeded(handle, output=b"late")


def _late_failed(queue, handle) -> None:
    queue.mark_failed(handle, error_message="late failure")


def _late_dead_lettered(queue, handle) -> None:
    queue.mark_dead_lettered(handle, reason=DeadLetterReason.RETRY_BUDGET_EXHAUSTED, error_message="late dl")


def _late_retry(queue, handle) -> None:
    returned = queue.retry_or_dead_letter(handle, error_message="late retry", retry_policy=backoff_policy(seconds=0.0))
    assert returned is queue.get_status(handle)  # reports the standing terminal status


_LATE = {
    "mark_succeeded": _late_succeeded,
    "mark_failed": _late_failed,
    "mark_dead_lettered": _late_dead_lettered,
    "retry_or_dead_letter": _late_retry,
}


@pytest.mark.parametrize("late", sorted(_LATE))
@pytest.mark.parametrize("terminal", list(_TERMINALS), ids=lambda status: status.value)
def test_terminal_status_is_never_overwritten(terminal: TaskStatus, late: str) -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    handle = _running(queue)
    _TERMINALS[terminal](queue, handle)
    before_status = queue.get_status(handle)
    before_result = queue.get_result(handle)
    assert before_status is terminal

    _LATE[late](queue, handle)

    assert queue.get_status(handle) is terminal
    assert queue.get_result(handle) == before_result  # output, error and dead-letter record untouched
    assert queue.claim_pending(tenant_id=handle.tenant_id, limit=10) == []  # nothing re-opened for work


def test_dead_letter_record_survives_a_late_success() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    handle = _running(queue)
    _to_dead_lettered(queue, handle)
    queue.mark_succeeded(handle, output=b"late")
    result = queue.get_result(handle)
    assert result is not None and result.status is TaskStatus.DEAD_LETTERED
    assert result.dead_letter is not None and result.dead_letter.last_error == "poison"


def test_running_task_still_reaches_every_terminal_status() -> None:
    for terminal, transition in _TERMINALS.items():
        queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
        handle = _running(queue)
        transition(queue, handle)
        assert queue.get_status(handle) is terminal


def test_retry_of_a_running_task_still_requeues_and_exhausts_into_the_dead_letter() -> None:
    queue = DocumentStoreTaskQueue(InMemoryDocumentStore())
    handle = _running(queue)
    policy = backoff_policy(max_retries=0, seconds=0.0)
    assert queue.retry_or_dead_letter(handle, error_message="x", retry_policy=policy) is TaskStatus.DEAD_LETTERED
    handle2 = _running(queue)
    again = backoff_policy(max_retries=3, seconds=0.0)
    assert queue.retry_or_dead_letter(handle2, error_message="x", retry_policy=again) is TaskStatus.PENDING
