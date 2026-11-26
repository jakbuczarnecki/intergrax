# © Artur Czarnecki. All rights reserved.

"""Optional Celery-backed run dispatch for Tier-3 fastapi_core hosts (H-APP-WIRING.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.production_queue_resolver import (
    ProductionQueueBackend,
    UnsupportedQueueBackendError,
    UnsupportedQueueBackendReason,
    broker_url_refused_in_strict_posture,
    production_queue_requires_worker,
    require_celery_broker_url_for_strict,
    resolve_production_queue_backend,
)
from intergrax.applications._shared.reliability_wiring import (
    assert_idempotency_store_satisfies_host_topology,
)
from intergrax.applications._shared.task_control_wiring import TaskEnricher
from intergrax.background_tasks.active_work import ActiveWorkSource
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore, DocumentStore
from intergrax.integrations.contracts.document_store_process_durability import (
    document_store_survives_process_restart,
)
from intergrax.integrations.providers.message_bus.celery.config import CeleryIntegrationConfig
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
    active_work_sources: tuple[ActiveWorkSource, ...] = ()


def _assert_durable_task_store(document_store: DocumentStore) -> ConditionalDocumentStore:
    if not isinstance(document_store, ConditionalDocumentStore):
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.TASK_STORE_NOT_DURABLE,
            "out-of-process queue execution requires ConditionalDocumentStore with CAS",
        )
    if not document_store_survives_process_restart(document_store):
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.TASK_STORE_NOT_DURABLE,
            "out-of-process queue execution requires a document store that survives process restart",
        )
    return document_store


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
    document_store: DocumentStore | None = None,
    idempotency_store: IdempotencyStore | None = None,
    required_idempotency_persistence_topology: PersistenceTopology | None = None,
    orchestration_triggers: frozenset[str] = frozenset(),
    pipeline_capability_suffix: str = ".pipeline",
    production_mode: bool,
) -> QueueWorkerWiring:
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

    if production_mode and document_store is not None and backend is ProductionQueueBackend.CELERY:
        _assert_durable_task_store(document_store)
        celery_config = CeleryIntegrationConfig.from_env(app_name=app_name)
        require_celery_broker_url_for_strict(broker_url=celery_config.broker_url)
        from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore

        if idempotency_store is None or isinstance(idempotency_store, InMemoryIdempotencyStore):
            raise UnsupportedQueueBackendError(
                UnsupportedQueueBackendReason.TASK_STORE_NOT_DURABLE,
                "out-of-process queue execution requires a durable idempotency ledger for the task fence",
            )
        if required_idempotency_persistence_topology is not None:
            assert_idempotency_store_satisfies_host_topology(
                idempotency_store=idempotency_store,
                required_topology=required_idempotency_persistence_topology,
            )
        if kv_store is None:
            raise ValueError(
                "wire_optional_queue_execution requires kv_store for Celery worker identity persistence",
            )
        if causal_evidence_persistence is None:
            raise ValueError(
                "wire_optional_queue_execution requires causal_evidence_persistence "
                "for required transport→execution audit evidence admission",
            )
        from intergrax.queueing.providers.document_store.document_store_task_queue import (
            DocumentStoreTaskQueue,
        )
        from intergrax.queueing.providers.document_store.durable_wake_task_queue import (
            DurableWakeTaskQueue,
        )
        from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

        durable_queue = DocumentStoreTaskQueue(
            document_store,
            idempotency_store=idempotency_store,
        )
        worker_app = create_nexus_celery_worker_app(
            app_name=celery_config.app_name,
            broker_url=celery_config.broker_url,
            backend_url=celery_config.backend_url,
            agent_registry=registry,
            host_execution=host_execution,
            task_always_eager=False,
            kv_store=kv_store,
            causal_evidence_persistence=causal_evidence_persistence,
            idempotency_store=idempotency_store,
            orchestration_triggers=orchestration_triggers,
            pipeline_capability_suffix=pipeline_capability_suffix,
            task_enricher=task_enricher,
            production_mode=production_mode,
            document_store_task_queue=durable_queue,
        )

        def _publish_wake(handle, request) -> None:  # noqa: ANN001
            worker_app.send_task(
                "intergrax.execute",
                kwargs={
                    "logical_task_name": request.task_name,
                    "tenant_id": request.tenant_id,
                    "run_id": request.run_id,
                    "payload": request.payload,
                    "idempotency_key": request.idempotency_key,
                    "document_task_id": handle.task_id,
                },
            )

        task_queue = DurableWakeTaskQueue(
            durable_queue=durable_queue,
            publish_wake=_publish_wake,
        )
        adapter = QueuedHostTaskExecutionAdapter(
            task_queue,
            run_service,
            wait_for_result=wait_for_result,
        )
        return QueueWorkerWiring(
            execution_adapter=adapter,
            celery_app=worker_app,
            worker_kind="out_of_process_durable",
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
    if broker_url_refused_in_strict_posture("memory://"):
        pass
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
