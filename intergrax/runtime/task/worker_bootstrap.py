# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Celery worker bootstrap with Nexus Task v2 handler (§41, J.3)."""

from __future__ import annotations  # noqa: I001

from collections.abc import Callable
from typing import Optional, Tuple  # noqa: F401, UP035

from celery import Celery

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.contracts.run_budget import RunBudget
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionPort,
)
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.execution.agent_runtime_context import (
    ArtifactStore,
    WorkerArtifactServices,
    compose_worker_artifact_services,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import (
    NexusWorkerRuntime,
    register_nexus_task_worker,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.worker_payload import NEXUS_TASK_V2_LOGICAL_NAME
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.durable_execution_wiring import (
    wire_durable_execution_runtime_dependencies,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def build_nexus_task_execution_registry(
    registry: AgentRegistry,
    *,
    host_execution: HostTaskExecutionPort | None = None,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity] | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None,
    lifecycle=None,
    kv_store: DistributedKVStore | None = None,
    run_budget: RunBudget | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    orchestration_triggers: frozenset[str] = frozenset(),
    pipeline_capability_suffix: str = ".pipeline",
    task_enricher=None,
    production_mode: bool,
    root_authority_admission: RootExecutionAuthorityAdmissionPort | None = None,
    artifact_store: ArtifactStore | None = None,
    artifact_governance: AgentRuntimeGovernancePort | None = None,
    artifact_audit: GovernanceAuditRecorder | None = None,
    on_artifact_services: Callable[[WorkerArtifactServices], None] | None = None,
) -> TaskExecutionRegistry:
    """Register ``nexus.task.v2`` on a worker TaskExecutionRegistry.

    With ``artifact_store`` the D8 retention sweep is registered on the same registry and the
    composed services (sweeper, governed read port) are handed to ``on_artifact_services``.
    """
    durable_deps = None
    if kv_store is not None:
        durable_deps = wire_durable_execution_runtime_dependencies(kv_store=kv_store)
    worker_registry = TaskExecutionRegistry()
    if host_execution is not None:
        runtime = NexusWorkerRuntime(
            host_execution,
            checkpoint_store=checkpoint_store,
            lifecycle=lifecycle,
            task_enricher=task_enricher,
        )
    elif admit_root_governance_identity is not None:
        if root_authority_admission is None:
            raise ValueError(
                "build_nexus_task_execution_registry requires root_authority_admission "
                "when constructing worker host execution from registry",
            )
        runtime = NexusWorkerRuntime.from_registry(
            registry,
            checkpoint_store=checkpoint_store,
            execution_continuation_state_store=execution_continuation_state_store,
            lifecycle=lifecycle,
            run_budget=run_budget,
            run_budget_persistence=(durable_deps.run_budget_persistence if durable_deps is not None else None),
            deadline_authority_resolver=(
                durable_deps.deadline_authority_resolver if durable_deps is not None else None
            ),
            execution_terminal=execution_terminal,
            orchestration_triggers=orchestration_triggers,
            pipeline_capability_suffix=pipeline_capability_suffix,
            task_enricher=task_enricher,
            admit_root_governance_identity=admit_root_governance_identity,
            root_authority_admission=root_authority_admission,
            production_mode=production_mode,
        )
    else:
        raise ValueError(
            "build_nexus_task_execution_registry requires host_execution or "
            "admit_root_governance_identity for governed worker admission",
        )
    register_nexus_task_worker(worker_registry, runtime)
    if artifact_store is not None:
        services = compose_worker_artifact_services(
            worker_registry,
            artifact_store=artifact_store,
            governance=artifact_governance,
            audit=artifact_audit,
        )
        if on_artifact_services is not None:
            on_artifact_services(services)
    return worker_registry


def create_nexus_celery_worker_app(
    *,
    app_name: str,
    broker_url: str,
    backend_url: str | None,
    agent_registry: AgentRegistry,
    host_execution: HostTaskExecutionPort | None = None,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity] | None = None,
    idempotency_store: IdempotencyStore | None = None,
    retry_policy: RetryPolicy | None = None,
    lock_ttl_seconds: int | None = None,
    completed_ttl_seconds: int | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None,
    rate_limiter: DistributedRateLimiter | None = None,
    rate_limit_config: Callable[[str], tuple[int, float]] | None = None,
    on_rate_limited: Callable[[RateLimitEvent], None] | None = None,
    on_retry_scheduled: Callable[[RetryEvent], None] | None = None,
    task_always_eager: bool = False,
    lifecycle=None,
    kv_store: DistributedKVStore | None = None,
    causal_evidence_persistence: CausalEvidencePersistence,
    orchestration_triggers: frozenset[str] = frozenset(),
    pipeline_capability_suffix: str = ".pipeline",
    task_enricher=None,
    production_mode: bool,
    root_authority_admission: RootExecutionAuthorityAdmissionPort | None = None,
    artifact_store: ArtifactStore | None = None,
    artifact_governance: AgentRuntimeGovernancePort | None = None,
    artifact_audit: GovernanceAuditRecorder | None = None,
    on_artifact_services: Callable[[WorkerArtifactServices], None] | None = None,
) -> Celery:
    """Production/lab composition root: Celery + ``nexus.task.v2`` handler."""
    if retry_policy is not None and lock_ttl_seconds is not None:
        max_retry_window = retry_policy.max_retry_window_seconds()
        if lock_ttl_seconds < max_retry_window:
            raise ValueError(
                "Invalid configuration: lock_ttl_seconds "
                f"({lock_ttl_seconds}) is smaller than maximum retry window "
                f"({max_retry_window})."
            )

    admission = wire_background_execution_admission_dependencies(kv_store=kv_store)

    worker_registry = build_nexus_task_execution_registry(
        agent_registry,
        host_execution=host_execution,
        admit_root_governance_identity=admit_root_governance_identity,
        checkpoint_store=checkpoint_store,
        execution_continuation_state_store=execution_continuation_state_store,
        lifecycle=lifecycle,
        kv_store=kv_store,
        execution_terminal=admission.execution_terminal,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        task_enricher=task_enricher,
        production_mode=production_mode,
        root_authority_admission=root_authority_admission,
        artifact_store=artifact_store,
        artifact_governance=artifact_governance,
        artifact_audit=artifact_audit,
        on_artifact_services=on_artifact_services,
    )

    app = Celery(app_name, broker=broker_url, backend=backend_url)
    app.conf.task_always_eager = task_always_eager
    app.conf.task_eager_propagates = task_always_eager
    if task_always_eager:
        app.conf.task_store_eager_result = True

    if kv_store is None:
        raise ValueError(
            "create_nexus_celery_worker_app requires kv_store for BG-EXEC-2 identity persistence",
        )

    register_dispatcher_task(
        app=app,
        registry=worker_registry,
        idempotency_store=idempotency_store,
        lock_ttl_seconds=lock_ttl_seconds,
        completed_ttl_seconds=completed_ttl_seconds,
        retry_policy=retry_policy,
        rate_limiter=rate_limiter,
        rate_limit_config=rate_limit_config,
        on_rate_limited=on_rate_limited,
        on_retry_scheduled=on_retry_scheduled,
        identity_persistence=admission.identity_persistence,
        causal_evidence_persistence=causal_evidence_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
    )

    return app


__all__ = [
    "NEXUS_TASK_V2_LOGICAL_NAME",
    "WorkerArtifactServices",
    "build_nexus_task_execution_registry",
    "compose_worker_artifact_services",
    "create_nexus_celery_worker_app",
]
