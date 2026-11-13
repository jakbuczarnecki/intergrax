# © Artur Czarnecki. All rights reserved.

"""PDC-D7-R009 / R013 — retry-budget exhaustion reaches a terminal DEAD_LETTERED state.

Real ``DocumentStoreTaskQueue`` + ``DocumentStoreTaskWorker`` over the canonical
background-execution admission wiring; handlers are real registry handlers.
"""

from __future__ import annotations

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import (
    DeadLetterReason,
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    is_terminal_task_status,
)
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.execution import RetryableHandlerError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel

pytestmark = [pytest.mark.unit]

TASK = "pdc.d7.task"
TENANT = "t-d7"


class _Out(BaseModel):
    ok: bool = True


def _policy(*, max_retries: int, backoff: float = 0.0) -> RetryPolicy:
    return RetryPolicy(
        max_retries=max_retries,
        initial_backoff_seconds=backoff,
        backoff_multiplier=1.0,
        max_backoff_seconds=None,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=True,
    )


class _Harness:
    def __init__(self, handler, *, retry_policy: RetryPolicy | None) -> None:
        self.store = InMemoryDocumentStore()
        self.queue = DocumentStoreTaskQueue(self.store)
        registry = TaskExecutionRegistry()
        registry.register(TASK, handler)
        admission = wire_background_execution_admission_dependencies(document_store=self.store)
        self.worker = DocumentStoreTaskWorker(
            self.queue,
            registry,
            identity_persistence=admission.identity_persistence,
            attempt_lifecycle=admission.attempt_lifecycle,
            execution_terminal=admission.execution_terminal,
            causal_evidence_persistence=wire_causal_evidence_persistence(
                document_store=self.store,
            ),
            retry_policy=retry_policy,
        )

    def enqueue(self) -> TaskHandle:
        return self.queue.enqueue(TaskRequest(tenant_id=TENANT, run_id="r-d7", task_name=TASK, payload=b"{}"))

    def drain(self, *, bound: int = 12) -> int:
        """Drain until idle (bounded); returns total claimed deliveries."""
        total = 0
        for _ in range(bound):
            claimed = self.worker.drain_once()
            if claimed == 0:
                return total
            total += claimed
        raise AssertionError("worker did not become idle within the drain bound")


def _poison_handler(calls: list[int]):
    def handler(*, tenant_id, run_id, payload, idempotency_key, execution_identity):
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        calls.append(1)
        raise RetryableHandlerError("poison payload always fails")

    return handler


def test_dead_lettered_is_a_terminal_state_of_the_queue_contract() -> None:
    assert TaskStatus("DEAD_LETTERED") is TaskStatus.DEAD_LETTERED
    assert is_terminal_task_status(TaskStatus.DEAD_LETTERED)
    assert is_terminal_task_status(TaskStatus.SUCCEEDED)
    assert is_terminal_task_status(TaskStatus.FAILED)
    assert not is_terminal_task_status(TaskStatus.PENDING)
    assert not is_terminal_task_status(TaskStatus.RUNNING)
    assert {reason.value for reason in DeadLetterReason} >= {"retry_budget_exhausted"}


def test_existing_provider_shape_stays_valid_without_dead_letter_support() -> None:
    class _MinimalProvider(TaskQueue):
        def enqueue(self, request: TaskRequest) -> TaskHandle:
            return TaskHandle(task_id="x", provider="minimal", tenant_id=request.tenant_id)

        def get_status(self, handle: TaskHandle) -> TaskStatus:
            return TaskStatus.PENDING

        def get_result(self, handle: TaskHandle) -> TaskResult | None:
            return None

    provider = _MinimalProvider()
    handle = provider.enqueue(TaskRequest(tenant_id="t", run_id="r", task_name="n", payload=b""))
    assert provider.get_status(handle) is TaskStatus.PENDING
    assert TaskResult(status=TaskStatus.FAILED).dead_letter is None  # additive field, default None


def test_retry_budget_exhaustion_reaches_terminal_dead_letter_with_reason_and_attempts() -> None:
    calls: list[int] = []
    harness = _Harness(_poison_handler(calls), retry_policy=_policy(max_retries=2))
    handle = harness.enqueue()

    assert harness.drain() == 3  # first delivery + exactly max_retries redeliveries
    assert len(calls) == 3
    assert harness.queue.get_status(handle) is TaskStatus.DEAD_LETTERED

    result = harness.queue.get_result(handle)
    assert result is not None
    assert result.status is TaskStatus.DEAD_LETTERED
    assert result.attempts == 3
    assert result.dead_letter is not None
    assert result.dead_letter.reason is DeadLetterReason.RETRY_BUDGET_EXHAUSTED
    assert result.dead_letter.attempts == 3
    assert "poison payload" in (result.dead_letter.last_error or "")

    # Terminal: nothing re-delivers it and it cannot be cancelled back to life.
    assert harness.drain() == 0
    assert len(calls) == 3
    assert harness.queue.cancel(handle) is False
    rows = harness.queue.list_tasks(TENANT, status_filter=TaskStatus.DEAD_LETTERED)
    assert [row.task_id for row in rows] == [handle.task_id]


def test_dead_letter_is_durable_across_queue_instances() -> None:
    harness = _Harness(_poison_handler([]), retry_policy=_policy(max_retries=0))
    handle = harness.enqueue()
    harness.drain()
    reopened = DocumentStoreTaskQueue(harness.store)
    result = reopened.get_result(handle)
    assert result is not None and result.dead_letter is not None
    assert result.dead_letter.attempts == 1


def test_zero_retry_budget_dead_letters_on_first_retryable_failure() -> None:
    calls: list[int] = []
    harness = _Harness(_poison_handler(calls), retry_policy=_policy(max_retries=0))
    handle = harness.enqueue()
    assert harness.drain() == 1
    assert len(calls) == 1
    assert harness.queue.get_status(handle) is TaskStatus.DEAD_LETTERED


def test_transient_failure_within_budget_succeeds_and_is_not_dead_lettered() -> None:
    calls: list[int] = []

    def flaky(*, tenant_id, run_id, payload, idempotency_key, execution_identity):
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        calls.append(1)
        if len(calls) == 1:
            raise RetryableHandlerError("transient")
        return ToolExecutionResult.ok(_Out())

    harness = _Harness(flaky, retry_policy=_policy(max_retries=3))
    handle = harness.enqueue()
    assert harness.drain() == 2
    assert harness.queue.get_status(handle) is TaskStatus.SUCCEEDED
    result = harness.queue.get_result(handle)
    assert result is not None and result.attempts == 2 and result.dead_letter is None


def test_non_retryable_failure_is_failed_not_dead_lettered() -> None:
    def broken(*, tenant_id, run_id, payload, idempotency_key, execution_identity):
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        raise ValueError("permanent bug")

    harness = _Harness(broken, retry_policy=_policy(max_retries=5))
    handle = harness.enqueue()
    assert harness.drain() == 1
    assert harness.queue.get_status(handle) is TaskStatus.FAILED


def test_without_a_retry_policy_failure_semantics_are_unchanged() -> None:
    calls: list[int] = []
    harness = _Harness(_poison_handler(calls), retry_policy=None)
    handle = harness.enqueue()
    assert harness.drain() == 1
    assert len(calls) == 1
    assert harness.queue.get_status(handle) is TaskStatus.FAILED


def test_backoff_defers_the_redelivery_until_it_is_due() -> None:
    calls: list[int] = []
    harness = _Harness(_poison_handler(calls), retry_policy=_policy(max_retries=2, backoff=3600.0))
    handle = harness.enqueue()
    assert harness.worker.drain_once() == 1
    assert harness.queue.get_status(handle) is TaskStatus.PENDING  # requeued, not terminal
    assert harness.worker.drain_once() == 0  # not due for an hour
    assert len(calls) == 1


def test_crash_looping_poison_task_is_dead_lettered_by_recovery() -> None:
    harness = _Harness(_poison_handler([]), retry_policy=None)
    handle = harness.enqueue()

    harness.queue.claim_pending(limit=1)  # delivery 1, host dies before settling
    harness.queue.recover_interrupted_running(max_retries=1)
    assert harness.queue.get_status(handle) is TaskStatus.PENDING  # within budget: redeliver

    harness.queue.claim_pending(limit=1)  # delivery 2, host dies again
    harness.queue.recover_interrupted_running(max_retries=1)
    assert harness.queue.get_status(handle) is TaskStatus.DEAD_LETTERED
    result = harness.queue.get_result(handle)
    assert result is not None and result.dead_letter is not None
    assert result.dead_letter.reason is DeadLetterReason.RETRY_BUDGET_EXHAUSTED
    assert result.dead_letter.attempts == 2


def test_recovery_without_a_budget_keeps_fail_closed_behaviour() -> None:
    harness = _Harness(_poison_handler([]), retry_policy=None)
    handle = harness.enqueue()
    harness.queue.claim_pending(limit=1)
    harness.queue.recover_interrupted_running()
    assert harness.queue.get_status(handle) is TaskStatus.FAILED
