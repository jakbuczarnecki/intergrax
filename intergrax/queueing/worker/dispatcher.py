# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from celery import Celery

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.execution import (
    execute_logical_task,
    IdempotencyLockConflictError,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy


def register_dispatcher_task(
    app: Celery,
    registry: TaskExecutionRegistry,
    kv_store: Optional[DistributedKVStore] = None,
    *,
    lock_ttl_seconds: Optional[int] = None,
    retry_policy: Optional[RetryPolicy] = None,
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

        except IdempotencyLockConflictError as exc:
            if (
                retry_policy is not None
                and retry_policy.retry_on_lock_conflict
            ):
                current_retries = self.request.retries

                if current_retries >= retry_policy.max_retries:
                    raise

                countdown = retry_policy.initial_backoff_seconds * (
                    retry_policy.backoff_multiplier ** current_retries
                )

                if (
                    retry_policy.max_backoff_seconds is not None
                    and countdown > retry_policy.max_backoff_seconds
                ):
                    countdown = retry_policy.max_backoff_seconds

                raise self.retry(exc=exc, countdown=countdown)

            raise