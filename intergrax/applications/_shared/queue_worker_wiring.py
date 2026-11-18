# © Artur Czarnecki. All rights reserved.

"""Optional Celery-backed run dispatch for Tier-3 fastapi_core hosts (H-APP-WIRING.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    UnsupportedQueueBackendError,
    UnsupportedQueueBackendReason,
    production_queue_requires_worker,
    resolve_production_queue_backend,
)
from intergrax.applications._shared.task_control_wiring import TaskEnricher
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.host_task_execution_run_adapter import (
    HostTaskExecutionRunAdapter,
)
from intergrax.runtime.task.queued_host_task_execution_adapter import (
    QueuedHostTaskExecutionAdapter,
)


@dataclass(frozen=True, slots=True)
class QueueWorkerWiring:
    """Resolved queue worker stack for a product host."""

    execution_adapter: ExecutionAdapter
    celery_app: object | None = None
    worker_kind: str = "inline"


def wire_optional_queue_execution(
    *,
    enabled: bool,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    run_service: DefaultRunService,
    task_enricher: TaskEnricher | None = None,
    wait_for_result: bool = True,
    app_name: str = "tier3_host_task_worker",
    queue_backend: ProductionQueueBackend | None = None,
    kv_store: DistributedKVStore | None = None,
    causal_evidence_persistence: CausalEvidencePersistence | None = None,
    orchestration_triggers: frozenset[str] = frozenset(),
    pipeline_capability_suffix: str = ".pipeline",
    production_mode: bool,
) -> QueueWorkerWiring:
    """
    Return inline host-task adapter or an in-process eager Celery queue adapter.

    The Celery worker built here is an in-process eager worker on a memory broker
    (``worker_kind == "eager_in_process_lab"``). It is refused when ``production_mode`` is
    true, and Kafka/RabbitMQ are never rewritten to it: no real-broker wiring exists yet
    (T-IGX-043-B, DP-30).
    """
    backend = queue_backend or resolve_production_queue_backend()
    if not enabled and not production_queue_requires_worker(backend):
        adapter: ExecutionAdapter = HostTaskExecutionRunAdapter(
            host_execution,
            task_enricher=task_enricher,
        )
        adapter.bind_run_service(run_service)
        return QueueWorkerWiring(execution_adapter=adapter)

    if backend in (ProductionQueueBackend.RABBITMQ, ProductionQueueBackend.KAFKA):
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.BROKER_WIRING_NOT_BUILT,
            f"queue backend {backend.value!r} has no Tier-3 broker wiring; "
            "it is not rewritten to the in-process eager Celery worker",
        )
    if production_mode:
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.EAGER_WORKER_FORBIDDEN_IN_PRODUCTION,
            "the Tier-3 queue worker is an in-process eager worker on a memory broker "
            "and is not allowed with production_mode=True",
        )

    from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
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
        host_execution=host_execution,
        task_always_eager=True,
        kv_store=kv_store,
        causal_evidence_persistence=causal_evidence_persistence,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        task_enricher=task_enricher,
        production_mode=production_mode,
    )
    queue = CeleryTaskQueue(worker_app)
    adapter = QueuedHostTaskExecutionAdapter(
        queue,
        run_service,
        wait_for_result=wait_for_result,
    )
    return QueueWorkerWiring(
        execution_adapter=adapter,
        celery_app=worker_app,
        worker_kind="eager_in_process_lab",
    )
