# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Callable, Optional

from celery import Celery

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.execution import (
    RetryableHandlerError,
    execute_logical_task,
    IdempotencyLockConflictError,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker.retry_backoff import calculate_retry_countdown


def register_dispatcher_task(
    app: Celery,
    registry: TaskExecutionRegistry,
    kv_store: Optional[DistributedKVStore] = None,
    *,
    lock_ttl_seconds: Optional[int] = None,
    retry_policy: Optional[RetryPolicy] = None,
    on_retry_scheduled: Optional[Callable[[RetryEvent], None]] = None,
) -> None:
    """
    Registers the generic execution task into Celery app.

    This must be called during worker composition phase.
    """

    if retry_policy is not None:
        retry_policy.validate()

    @app.task(name="intergrax.execute", bind=True)
    def intergrax_execute(
        self,
        logical_task_name: str,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str] = None,
    ) -> bytes:
        """
        Generic execution entrypoint.

        Dispatches execution to registered logical task handler.
        """

        try:
            return execute_logical_task(
                registry=registry,
                logical_task_name=logical_task_name,
                tenant_id=tenant_id,
                run_id=run_id,
                payload=payload,
                idempotency_key=idempotency_key,
                kv_store=kv_store,
                lock_ttl_seconds=lock_ttl_seconds,
            )

        except (IdempotencyLockConflictError, RetryableHandlerError) as exc:

            if retry_policy is None:
                raise

            # Determine retry eligibility based on exception type
            if isinstance(exc, IdempotencyLockConflictError):
                should_retry = retry_policy.retry_on_lock_conflict
            else:
                should_retry = retry_policy.retry_on_handler_exception

            if not should_retry:
                raise

            current_retries: int = self.request.retries

            if current_retries >= retry_policy.max_retries:
                raise

            countdown: float = calculate_retry_countdown(
                policy=retry_policy,
                current_retries=current_retries,
            )

            reason: str = (
                "lock_conflict"
                if isinstance(exc, IdempotencyLockConflictError)
                else "handler_transient"
            )

            event = RetryEvent(
                logical_task_name=logical_task_name,
                exception_type=type(exc).__name__,
                current_retries=current_retries,
                max_retries=retry_policy.max_retries,
                countdown_seconds=countdown,
                reason=reason,
            )

            if on_retry_scheduled is not None:
                on_retry_scheduled(event)

            raise self.retry(exc=exc, countdown=countdown)