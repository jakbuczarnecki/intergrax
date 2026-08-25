# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pydantic import BaseModel
import pytest
from celery import Celery
from celery.exceptions import Retry

from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.runtime.background_execution.identity_persistence import (
    wire_background_execution_identity_persistence,
)
from tests.unit.queueing.worker.dispatcher_test_kv import DispatcherTestKVStore
from intergrax.queueing.worker.execution import RetryableHandlerError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.unit

class _DummyOutput(BaseModel):
    value: str


class FakeRegistry(TaskExecutionRegistry):
    def get_handler(self, logical_task_name: str):
        def handler(
            *,
            tenant_id: str,
            run_id: str,
            payload: bytes,
            idempotency_key,
        ) -> ToolExecutionResult[_DummyOutput]:
            raise RetryableHandlerError("transient failure")

        return handler


@pytest.mark.unit
def test_dispatcher_calls_retry_hook() -> None:
    app = Celery("test")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True

    registry = FakeRegistry()

    retry_policy = RetryPolicy(
        max_retries=3,
        initial_backoff_seconds=5.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=None,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=True,
    )

    captured_event: list[RetryEvent] = []

    def retry_hook(event: RetryEvent) -> None:
        captured_event.append(event)

    register_dispatcher_task(
        app=app,
        registry=registry,
        idempotency_store=None,
        lock_ttl_seconds=None,
        retry_policy=retry_policy,
        on_retry_scheduled=retry_hook,
        identity_persistence=wire_background_execution_identity_persistence(
            kv_store=DispatcherTestKVStore(),
        ),
    )

    task = app.tasks["intergrax.execute"]

    with pytest.raises(Retry):
        task.apply(
            kwargs={
                "logical_task_name": "test_task",
                "tenant_id": "tenant",
                "run_id": "run",
                "payload": b"data",
                "idempotency_key": None,
            }
        )

    assert len(captured_event) == 1

    event = captured_event[0]

    assert event.logical_task_name == "test_task"
    assert event.exception_type == "RetryableHandlerError"
    assert event.current_retries == 0
    assert event.max_retries == 3
    assert event.countdown_seconds == 5.0
    assert event.reason == "handler_transient"