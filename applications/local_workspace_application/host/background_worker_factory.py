# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker composition root (LKW.4E)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime, build_harness_host_runtime
from intergrax.applications._shared.host_queue_execution_wiring import (
    resolve_host_queue_execution_dependencies,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.task_control_wiring import (
    TaskEnricher,
    build_reliability_task_enricher,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.contracts.execution_identity import AttemptId, RunId
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.task.task import Task, TaskResult as RuntimeTaskResult
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
)
from local_workspace_application.background_ingest.worker_handler import (
    make_background_ingest_worker_handler,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.execution_wiring import build_lkw_host_task_execution
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)


@dataclass(frozen=True, slots=True)
class _BackgroundIngestHostExecutionRunner:
    _host_execution: HostTaskExecution
    _task_enricher: TaskEnricher

    async def run_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
    ) -> RuntimeTaskResult:
        prepared = self._task_enricher(task) if self._task_enricher is not None else task
        return await self._host_execution.execute(
            prepared,
            run_id=run_id,
            attempt_id=attempt_id,
        )


@dataclass(frozen=True, slots=True)
class LocalWorkspaceBackgroundWorkerWiring:
    runtime: HarnessHostRuntime
    host_execution: HostTaskExecution
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
    host_execution = runtime.execution
    runner = _BackgroundIngestHostExecutionRunner(
        _host_execution=host_execution,
        _task_enricher=task_enricher,
    )

    registry = TaskExecutionRegistry()
    task_registry = TaskRegistry()
    handler = make_background_ingest_worker_handler(runner)
    task_registry.register(
        TaskDefinition(
            task_name=LKW_BACKGROUND_INGEST_TASK_NAME,
            payload_schema=LkwBackgroundIngestJob,
            handler=handler,
            provider="kafka",
        )
    )
    task_registry.bind_execution_registry(registry)

    queue_dependencies = resolve_host_queue_execution_dependencies(runtime)
    worker = create_kafka_worker(
        kv_store=queue_dependencies.kv_store,
        execution_registry=registry,
        idempotency_store=runtime.reliability.idempotency_store,
        causal_evidence_persistence=queue_dependencies.causal_evidence_persistence,
    )
    return LocalWorkspaceBackgroundWorkerWiring(
        runtime=runtime,
        host_execution=host_execution,
        registry=registry,
        kv_store=queue_dependencies.kv_store,
        idempotency_store=runtime.reliability.idempotency_store,
        worker=worker,
    )
