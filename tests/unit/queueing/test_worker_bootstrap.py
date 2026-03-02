# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from celery import Celery
from pydantic import BaseModel
import pytest

from intergrax.queueing.worker_bootstrap import create_celery_worker_app
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = pytest.mark.unit


class _DummyOutput(BaseModel):
    value: str


class DummyRegistry(TaskExecutionRegistry):
    def get_handler(self, logical_task_name: str):
        def handler(
            *,
            tenant_id: str,
            run_id: str,
            payload: bytes,
            idempotency_key,
        ) -> ToolExecutionResult[_DummyOutput]:
            return ToolExecutionResult(
                success=True,
                output=_DummyOutput(value="ok"),
                error=None,
            )

        return handler


def test_create_celery_worker_app_registers_dispatcher() -> None:
    registry = DummyRegistry()

    retry_policy = RetryPolicy(
        max_retries=3,
        initial_backoff_seconds=1.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=10.0,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=True,
        jitter=False,
    )

    app: Celery = create_celery_worker_app(
        app_name="test_app",
        broker_url="memory://",
        backend_url=None,
        registry=registry,
        idempotency_store=None,
        retry_policy=retry_policy,
        lock_ttl_seconds=None,
        completed_ttl_seconds=None,
    )

    assert isinstance(app, Celery)
    assert "intergrax.execute" in app.tasks