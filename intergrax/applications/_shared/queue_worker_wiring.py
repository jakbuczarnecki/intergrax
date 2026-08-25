# © Artur Czarnecki. All rights reserved.

"""Optional Celery-backed run dispatch for Tier-3 fastapi_core hosts (H-APP-WIRING.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    production_queue_requires_worker,
    resolve_production_queue_backend,
)
from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


@dataclass(frozen=True, slots=True)
class QueueWorkerWiring:
    """Resolved queue worker stack for a product host."""

    execution_adapter: ExecutionAdapter
    celery_app: object | None = None


def wire_optional_queue_execution(
    *,
    enabled: bool,
    registry: AgentRegistry,
    task_runner: UnifiedTaskRunner,
    run_service: DefaultRunService,
    wait_for_result: bool = True,
    app_name: str = "tier3_nexus_worker",
    queue_backend: ProductionQueueBackend | None = None,
    kv_store: DistributedKVStore | None = None,
    causal_evidence_persistence: CausalEvidencePersistence | None = None,
) -> QueueWorkerWiring:
    """
    Return inline Nexus adapter or Celery queue adapter.

    ``wait_for_result=True`` uses eager Celery — suitable for gate tests and single-process deploys.
    """
    backend = queue_backend or resolve_production_queue_backend()
    if not enabled and not production_queue_requires_worker(backend):
        adapter: ExecutionAdapter = NexusTaskExecutionAdapter(task_runner)
        return QueueWorkerWiring(execution_adapter=adapter)

    if backend in (ProductionQueueBackend.RABBITMQ, ProductionQueueBackend.KAFKA):
        # Broker transports require external infra; fall back to eager Celery for harness hosts.
        backend = ProductionQueueBackend.CELERY

    from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
    from intergrax.runtime.task.queued_nexus_execution_adapter import QueuedNexusExecutionAdapter
    from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

    if kv_store is None:
        raise ValueError(
            "wire_optional_queue_execution requires kv_store for Celery worker identity persistence",
        )
    if causal_evidence_persistence is None:
        raise ValueError(
            "wire_optional_queue_execution requires causal_evidence_persistence "
            "for required transport→execution audit evidence admission",
        )

    worker_app = create_nexus_celery_worker_app(
        app_name=app_name,
        broker_url="memory://",
        backend_url="cache+memory://",
        agent_registry=registry,
        task_always_eager=True,
        kv_store=kv_store,
        causal_evidence_persistence=causal_evidence_persistence,
    )
    queue = CeleryTaskQueue(worker_app)
    adapter = QueuedNexusExecutionAdapter(
        queue,
        run_service,
        wait_for_result=wait_for_result,
    )
    return QueueWorkerWiring(execution_adapter=adapter, celery_app=worker_app)
