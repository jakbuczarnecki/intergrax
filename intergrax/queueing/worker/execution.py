# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.registry import TaskExecutionRegistry


def execute_logical_task(
    *,
    registry: TaskExecutionRegistry,
    logical_task_name: str,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: Optional[str],
    kv_store: Optional[DistributedKVStore],
    lock_ttl_seconds: Optional[int],
) -> bytes:
    """
    Pure execution core for logical task dispatch.

    Contains idempotency logic and handler invocation.
    Does not depend on Celery.
    """

    handler = registry.get_handler(logical_task_name)

    key: Optional[str] = None

    if idempotency_key and kv_store is not None:
        if lock_ttl_seconds is None:
            raise ValueError(
                "lock_ttl_seconds must be provided when kv_store is used."
            )

        key = f"idempotency:{tenant_id}:{idempotency_key}"

        acquired = kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=key,
            expected=None,
            new_value=b"__LOCK__",
            ttl_seconds=lock_ttl_seconds,
        )

        if not acquired:
            existing = kv_store.get(
                tenant_id=tenant_id,
                key=key,
            )
            if existing is not None and existing != b"__LOCK__":
                return existing

            raise RuntimeError(
                f"Idempotency lock is held for key '{idempotency_key}'."
            )

    result = handler(
        tenant_id=tenant_id,
        run_id=run_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )

    if key is not None and kv_store is not None:
        kv_store.set(
            tenant_id=tenant_id,
            key=key,
            value=result,
            ttl_seconds=lock_ttl_seconds,
        )

    return result