# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Celery worker bootstrap with Nexus Task v2 handler (§41, J.3)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from celery import Celery

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.queueing.worker.dispatcher import register_dispatcher_task
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import (
    NexusWorkerRuntime,
    register_nexus_task_worker,
)
from intergrax.runtime.task.worker_payload import NEXUS_TASK_V2_LOGICAL_NAME
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.budget.persistence import wire_run_budget_persistence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


def build_nexus_task_execution_registry(
    registry: AgentRegistry,
    *,
    checkpoint_store: Optional[TaskCheckpointPersistence] = None,
    lifecycle=None,
    kv_store: Optional[DistributedKVStore] = None,
    run_budget: RunBudget | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
) -> TaskExecutionRegistry:
    """Register ``nexus.task.v2`` on a worker TaskExecutionRegistry."""
    run_budget_persistence = None
    if kv_store is not None:
        run_budget_persistence = wire_run_budget_persistence(kv_store=kv_store)
    worker_registry = TaskExecutionRegistry()
    runtime = NexusWorkerRuntime.from_registry(
        registry,
        checkpoint_store=checkpoint_store,
        lifecycle=lifecycle,
        run_budget=run_budget,
        run_budget_persistence=run_budget_persistence,
        execution_terminal=execution_terminal,
    )
    register_nexus_task_worker(worker_registry, runtime)
    return worker_registry


def create_nexus_celery_worker_app(
    *,
    app_name: str,
    broker_url: str,
    backend_url: Optional[str],
    agent_registry: AgentRegistry,
    idempotency_store: Optional[IdempotencyStore] = None,
    retry_policy: Optional[RetryPolicy] = None,
    lock_ttl_seconds: Optional[int] = None,
    completed_ttl_seconds: Optional[int] = None,
    checkpoint_store: Optional[TaskCheckpointPersistence] = None,
    rate_limiter: Optional[DistributedRateLimiter] = None,
    rate_limit_config: Optional[Callable[[str], Tuple[int, float]]] = None,
    on_rate_limited: Optional[Callable[[RateLimitEvent], None]] = None,
    on_retry_scheduled: Optional[Callable[[RetryEvent], None]] = None,
    task_always_eager: bool = False,
    lifecycle=None,
    kv_store: Optional[DistributedKVStore] = None,
    causal_evidence_persistence: CausalEvidencePersistence,
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
        checkpoint_store=checkpoint_store,
        lifecycle=lifecycle,
        kv_store=kv_store,
        execution_terminal=admission.execution_terminal,
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
    "build_nexus_task_execution_registry",
    "create_nexus_celery_worker_app",
]
