# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from celery import Celery

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy


def create_celery_worker_app(
    *,
    app_name: str,
    broker_url: str,
    backend_url: Optional[str],
    registry: TaskExecutionRegistry,
    kv_store: Optional[DistributedKVStore],
    retry_policy: Optional[RetryPolicy],
    lock_ttl_seconds: Optional[int],
) -> Celery:
    """
    Production composition root for Tier-0 execution plane.

    Creates Celery app and registers dispatcher.
    """

    app = Celery(app_name, broker=broker_url, backend=backend_url)

    register_dispatcher_task(
        app=app,
        registry=registry,
        kv_store=kv_store,
        lock_ttl_seconds=lock_ttl_seconds,
        retry_policy=retry_policy,
    )

    return app