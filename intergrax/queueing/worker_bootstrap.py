# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Callable, Optional, Tuple

from celery import Celery

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.background_execution.identity_persistence import (
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def create_celery_worker_app(
    *,
    app_name: str,
    broker_url: str,
    backend_url: Optional[str],
    registry: TaskExecutionRegistry,
    idempotency_store: Optional[IdempotencyStore],
    retry_policy: Optional[RetryPolicy],
    lock_ttl_seconds: Optional[int],
    completed_ttl_seconds: Optional[int],
    rate_limiter: Optional[DistributedRateLimiter] = None,
    rate_limit_config: Optional[Callable[[str], Tuple[int, float]]] = None,
    on_rate_limited: Optional[Callable[[RateLimitEvent], None]] = None,
    on_retry_scheduled: Optional[Callable[[RetryEvent], None]] = None,
    kv_store: Optional[DistributedKVStore] = None,
    causal_evidence_persistence: CausalEvidencePersistence,
) -> Celery:
    """
    Production composition root for Tier-0 execution plane.

    Creates Celery app and registers dispatcher.
    """

    # ------------------------------------------------------------------
    # Production safety validation: lease vs retry window
    # ------------------------------------------------------------------

    if retry_policy is not None and lock_ttl_seconds is not None:
        max_retry_window: float = retry_policy.max_retry_window_seconds()

        if lock_ttl_seconds < max_retry_window:
            raise ValueError(
                "Invalid configuration: lock_ttl_seconds "
                f"({lock_ttl_seconds}) is smaller than maximum retry window "
                f"({max_retry_window}). This may cause double execution."
            )

    app = Celery(app_name, broker=broker_url, backend=backend_url)

    if kv_store is None:
        raise ValueError(
            "create_celery_worker_app requires kv_store for BG-EXEC-2 identity persistence",
        )

    register_dispatcher_task(
        app=app,
        registry=registry,
        idempotency_store=idempotency_store,
        lock_ttl_seconds=lock_ttl_seconds,
        completed_ttl_seconds=completed_ttl_seconds,
        retry_policy=retry_policy,
        rate_limiter=rate_limiter,
        rate_limit_config=rate_limit_config,
        on_rate_limited=on_rate_limited,
        on_retry_scheduled=on_retry_scheduled,
        identity_persistence=wire_background_execution_identity_persistence(
            kv_store=kv_store,
        ),
        causal_evidence_persistence=causal_evidence_persistence,
    )

    return app