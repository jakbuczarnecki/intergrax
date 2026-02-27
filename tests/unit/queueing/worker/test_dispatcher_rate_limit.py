# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Tuple
from unittest.mock import Mock, patch

import pytest

from celery import Celery

from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy

pytestmark = pytest.mark.unit


class DenyRateLimiter(DistributedRateLimiter):
    def acquire(
        self,
        *,
        tenant_id: str,
        key: str,
        capacity: int,
        refill_rate_per_second: float,
    ) -> RateLimitResult:
        return RateLimitResult(
            allowed=False,
            remaining_tokens=0.0,
            retry_after_seconds=3.5,
        )


def _create_retry_policy(max_retries: int) -> RetryPolicy:
    return RetryPolicy(
        max_retries=max_retries,
        initial_backoff_seconds=1.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=None,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=True,
    )


def test_rate_limited_triggers_retry() -> None:
    app = Celery("test_app", broker="memory://", backend="rpc://")

    registry = TaskExecutionRegistry()
    retry_policy = _create_retry_policy(max_retries=5)

    rate_limiter = DenyRateLimiter()

    def rate_limit_config(_: str) -> Tuple[int, float]:
        return (10, 1.0)

    register_dispatcher_task(
        app=app,
        registry=registry,
        kv_store=None,
        rate_limiter=rate_limiter,
        retry_policy=retry_policy,
        rate_limit_config=rate_limit_config,
    )

    task = app.tasks["intergrax.execute"]

    retry_mock = Mock(side_effect=Exception("retry_called"))
    task.retry = retry_mock  # type: ignore

    with pytest.raises(Exception, match="retry_called"):
        task.run(
            logical_task_name="test_task",
            tenant_id="tenant_A",
            run_id="run_1",
            payload=b"{}",
            idempotency_key=None,
        )

    retry_mock.assert_called_once()
    _, kwargs = retry_mock.call_args
    assert 3.5 <= kwargs["countdown"] <= 4.2



def test_rate_limited_retry_applies_jitter() -> None:
    app = Celery("test_app_jitter", broker="memory://", backend="rpc://")

    registry = TaskExecutionRegistry()
    retry_policy = _create_retry_policy(max_retries=5)

    rate_limiter = DenyRateLimiter()

    def rate_limit_config(_: str) -> Tuple[int, float]:
        return (10, 1.0)

    register_dispatcher_task(
        app=app,
        registry=registry,
        kv_store=None,
        rate_limiter=rate_limiter,
        retry_policy=retry_policy,
        rate_limit_config=rate_limit_config,
    )

    task = app.tasks["intergrax.execute"]

    retry_mock = Mock(side_effect=Exception("retry_called"))
    task.retry = retry_mock  # type: ignore

    # retry_after_seconds = 3.5
    # jitter_window = 3.5 * 0.2 = 0.7
    # we force jitter = 0.5

    with patch("intergrax.queueing.worker.dispatcher.random.uniform", return_value=0.5):
        with pytest.raises(Exception, match="retry_called"):
            task.run(
                logical_task_name="test_task",
                tenant_id="tenant_A",
                run_id="run_1",
                payload=b"{}",
                idempotency_key=None,
            )

    _, kwargs = retry_mock.call_args
    assert kwargs["countdown"] == pytest.approx(4.0)