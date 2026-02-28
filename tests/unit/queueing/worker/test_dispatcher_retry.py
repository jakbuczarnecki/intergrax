# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from celery import Celery
from celery.exceptions import Retry

from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.execution import IdempotencyLockConflictError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy

pytestmark = pytest.mark.unit


class FakeRegistry(TaskExecutionRegistry):
    def get_handler(self, logical_task_name: str):
        def handler(*, tenant_id: str, run_id: str, payload: bytes, idempotency_key):
            raise IdempotencyLockConflictError("lock held")

        return handler


@pytest.mark.unit
def test_dispatcher_retries_on_lock_conflict() -> None:
    app = Celery("test")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True

    registry = FakeRegistry()

    retry_policy = RetryPolicy(
        max_retries=3,
        initial_backoff_seconds=2.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=10.0,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=False,
        jitter=False,
    )

    register_dispatcher_task(
        app=app,
        registry=registry,
        idempotency_store=None,
        lock_ttl_seconds=None,
        completed_ttl_seconds=None,
        retry_policy=retry_policy,
    )

    task = app.tasks["intergrax.execute"]

    with pytest.raises(Retry) as exc_info:
        task.apply(
            kwargs={
                "logical_task_name": "test",
                "tenant_id": "tenant",
                "run_id": "run",
                "payload": b"data",
                "idempotency_key": "key",
            }
        )

    retry_exc = exc_info.value

    # First retry → retries == 0 → countdown = initial_backoff
    assert retry_exc.when == 2.0