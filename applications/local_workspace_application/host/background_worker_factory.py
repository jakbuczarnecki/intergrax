# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker composition root (LKW.4E)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime, build_harness_host_runtime
from intergrax.applications._shared.task_control_wiring import (
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_worker
from intergrax.integrations.providers.key_value_cache.redis.bundle import create_redis_kv_store
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from local_workspace_application.background_ingest.worker_handler import (
    register_background_ingest_worker_handler,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


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
    settings: LocalWorkspaceBackendSettings | None = None,
) -> LocalWorkspaceBackgroundWorkerWiring:
    settings = settings or LocalWorkspaceBackendSettings.from_env()
    environment = manifest.resolved_environment()
    runtime = build_harness_host_runtime(manifest, environment, settings=settings)
    task_enricher = build_reliability_task_enricher(
        environment,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
        compensation_queue_store=runtime.compensation_queue_store,
        idempotency_store=runtime.reliability.idempotency_store,
    )
    task_runner = build_task_runner_with_enricher(runtime.nexus_loop, task_enricher)

    registry = TaskExecutionRegistry()
    register_background_ingest_worker_handler(registry, task_runner)

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
