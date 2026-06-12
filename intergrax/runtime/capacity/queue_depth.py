# © Artur Czarnecki. All rights reserved.

"""Queue depth signal provider (ECP-PROD.1)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.task_index import load_task_index


def pending_queue_depth(
    kv_store: DistributedKVStore,
    tenant_id: str,
    *,
    provider: str = "celery",
) -> float:
    """Count pending tasks for a broker provider in the task index."""
    depth = 0
    for record in load_task_index(kv_store, tenant_id):
        if record.provider != provider:
            continue
        if record.status == TaskStatus.PENDING.value:
            depth += 1
    return float(depth)


def make_queue_depth_provider(
    kv_store: DistributedKVStore,
    tenant_id: str,
    *,
    provider: str = "celery",
) -> Callable[[], float]:
    """Factory for CapacitySignalCollector.queue_depth_provider."""

    def _provider() -> float:
        return pending_queue_depth(kv_store, tenant_id, provider=provider)

    return _provider
