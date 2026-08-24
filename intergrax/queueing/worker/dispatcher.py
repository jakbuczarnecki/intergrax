# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

from celery import Celery

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.rate_limiter import (
    DistributedRateLimiter,
    RateLimitResult,
)
from intergrax.queueing.worker.execution import (
    RetryableHandlerError,
    execute_logical_task,
    IdempotencyLockConflictError,
)
from intergrax.queueing.worker.result_codec import encode_logical_task_result
from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker.retry_backoff import calculate_retry_countdown


def register_dispatcher_task(
    app: Celery,
    registry: TaskExecutionRegistry,
    idempotency_store: Optional[IdempotencyStore] = None,
    rate_limiter: Optional[DistributedRateLimiter] = None,
    *,
    lock_ttl_seconds: Optional[int] = None,
    completed_ttl_seconds: Optional[int] = None,
    retry_policy: Optional[RetryPolicy] = None,
    on_retry_scheduled: Optional[Callable[[RetryEvent], None]] = None,
    rate_limit_config: Optional[Callable[[str], Tuple[int, float]]] = None,
    on_rate_limited: Optional[Callable[[RateLimitEvent], None]] = None,
) -> None:
    """
    Registers the generic execution task into Celery app.

    This must be called during worker composition phase.
    """

    if retry_policy is not None:
        retry_policy.validate()

    if rate_limiter is not None and retry_policy is None:
        raise ValueError("retry_policy must be provided when rate_limiter is enabled.")

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

        # -------------------------
        # Tier-0: Distributed Rate Limiting
        # -------------------------
        if rate_limiter is not None:
            if rate_limit_config is None:
                raise ValueError(
                    "rate_limit_config must be provided when rate_limiter is enabled."
                )

            capacity, refill_rate_per_second = rate_limit_config(logical_task_name)

            result: RateLimitResult = rate_limiter.acquire(
                tenant_id=tenant_id,
                key=logical_task_name,
                capacity=int(capacity),
                refill_rate_per_second=float(refill_rate_per_second),
            )

            if not result.allowed:
                current_retries: int = self.request.retries

                if current_retries >= retry_policy.max_retries:
                    raise RuntimeError("rate_limited: max_retries reached")

                base_countdown: float = float(result.retry_after_seconds)

                # ---- RateLimit hook (attached to task, not closure) ----
                rate_limit_hook: Optional[
                    Callable[[RateLimitEvent], None]
                ] = self.on_rate_limited  # type: ignore[attr-defined]

                if rate_limit_hook is not None:
                    rate_limit_event = RateLimitEvent(
                        logical_task_name=logical_task_name,
                        tenant_id=tenant_id,
                        retry_after_seconds=base_countdown,
                        remaining_tokens=result.remaining_tokens,
                        current_retries=current_retries,
                        max_retries=retry_policy.max_retries,
                    )
                    rate_limit_hook(rate_limit_event)

                # ---- Jitter (20%) ----
                jitter_window: float = base_countdown * 0.2
                jitter: float = random.uniform(0.0, jitter_window)
                countdown: float = base_countdown + jitter

                # ---- Retry hook ----
                retry_hook: Optional[
                    Callable[[RetryEvent], None]
                ] = self.on_retry_scheduled  # type: ignore[attr-defined]

                if retry_hook is not None:
                    retry_event = RetryEvent(
                        logical_task_name=logical_task_name,
                        exception_type="RateLimited",
                        current_retries=current_retries,
                        max_retries=retry_policy.max_retries,
                        countdown_seconds=countdown,
                        reason="rate_limited",
                    )
                    retry_hook(retry_event)

                raise self.retry(
                    exc=RuntimeError("rate_limited"),
                    countdown=countdown,
                )

        # -------------------------
        # Core Execution
        # -------------------------
        try:
            execution_identity = bootstrap_background_execution(
                transport_tenant_id=tenant_id,
            )
            return encode_logical_task_result(
                execute_logical_task(
                    registry=registry,
                    logical_task_name=logical_task_name,
                    tenant_id=execution_identity.tenant_id,
                    run_id=str(execution_identity.run_id),
                    payload=payload,
                    idempotency_key=idempotency_key,
                    idempotency_store=idempotency_store,
                    lease_seconds=lock_ttl_seconds,
                    completed_ttl_seconds=completed_ttl_seconds,
                    execution_identity=execution_identity,
                )
            )

        except (IdempotencyLockConflictError, RetryableHandlerError) as exc:

            if retry_policy is None:
                raise

            if isinstance(exc, IdempotencyLockConflictError):
                should_retry = retry_policy.retry_on_lock_conflict
                reason = "lock_conflict"
            else:
                should_retry = retry_policy.retry_on_handler_exception
                reason = "handler_transient"

            if not should_retry:
                raise

            current_retries: int = self.request.retries

            if current_retries >= retry_policy.max_retries:
                raise

            countdown: float = calculate_retry_countdown(
                policy=retry_policy,
                current_retries=current_retries,
            )

            retry_hook: Optional[
                Callable[[RetryEvent], None]
            ] = self.on_retry_scheduled  # type: ignore[attr-defined]

            if retry_hook is not None:
                retry_event = RetryEvent(
                    logical_task_name=logical_task_name,
                    exception_type=type(exc).__name__,
                    current_retries=current_retries,
                    max_retries=retry_policy.max_retries,
                    countdown_seconds=countdown,
                    reason=reason,
                )
                retry_hook(retry_event)

            raise self.retry(exc=exc, countdown=countdown)

    # ---------------------------------------------
    # Attach hooks to task instance (production-safe)
    # ---------------------------------------------
    intergrax_execute.on_rate_limited = on_rate_limited  # type: ignore[attr-defined]
    intergrax_execute.on_retry_scheduled = on_retry_scheduled  # type: ignore[attr-defined]