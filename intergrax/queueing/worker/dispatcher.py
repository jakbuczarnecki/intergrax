# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from celery import Celery

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry


def register_dispatcher_task(
    app: Celery,
    registry: TaskExecutionRegistry,
    kv_store: Optional[DistributedKVStore] = None,
    *,
    lock_ttl_seconds: Optional[int] = None,
) -> None:
    """
    Registers the generic execution task into Celery app.

    This must be called during worker composition phase.
    """

    @app.task(name="intergrax.execute")
    def intergrax_execute(
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