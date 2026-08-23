# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker composition root (LKW.4E)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime, build_harness_host_runtime
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker
from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_kv_store
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
)
from local_workspace_application.background_ingest.worker_handler import (
    make_background_ingest_worker_handler,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)


@dataclass(frozen=True, slots=True)
class LocalWorkspaceBackgroundWorkerWiring:
    runtime: HarnessHostRuntime
    task_runner: UnifiedTaskRunner
    registry: TaskExecutionRegistry
    kv_store: DistributedKVStore
    idempotency_store: IdempotencyStore | None
    worker: object


def build_local_workspace_background_worker_wiring(
    *,
    manifest: ApplicationManifest,
    registry_projection: MaterializedRegistryProjection,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> LocalWorkspaceBackgroundWorkerWiring:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    environment = manifest.resolved_environment()
    lkw_document_store = resolve_lkw_runtime_document_store(settings)
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        settings=settings,
        idempotency_db_path=Path(settings.idempotency_db_path),
        document_store=lkw_document_store,
        registry_projection=registry_projection,
    )
    task_enricher = build_reliability_task_enricher(
        environment,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(runtime.nexus_loop, task_enricher)

    registry = TaskExecutionRegistry()
    task_registry = TaskRegistry()
    handler = make_background_ingest_worker_handler(task_runner)
    task_registry.register(
        TaskDefinition(
            task_name=LKW_BACKGROUND_INGEST_TASK_NAME,
            payload_schema=LkwBackgroundIngestJob,
            handler=handler,
            provider="kafka",
        )
    )
    task_registry.bind_execution_registry(registry)

    kv_store = create_redis_kv_store()
    worker = create_kafka_worker(
        kv_store=kv_store,
        execution_registry=registry,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    return LocalWorkspaceBackgroundWorkerWiring(
        runtime=runtime,
        task_runner=task_runner,
        registry=registry,
        kv_store=kv_store,
        idempotency_store=runtime.reliability.idempotency_store,
        worker=worker,
    )
