# © Artur Czarnecki. All rights reserved.

"""Tier-3 reliability wiring (Phase REL-1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.compensation_wiring import resolve_compensation_flow
from intergrax.applications._shared.partial_results_wiring import apply_partial_results_task_defaults
from intergrax.applications._shared.reasoning_wiring import resolve_replan_policy_context
from intergrax.applications._shared.reliability_runtime_bridge import (
    ReliabilityWiringOptions,
    resolve_reliability_wiring_options,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.integrations._shared.circuit_breaker import IntegrationCircuitBreakerConfig
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.applications._shared.autonomy_middleware import AutonomyGovernanceMiddleware
from intergrax.applications._shared.application_security_wiring import _attach_middleware
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class ApplicationReliabilityWiring:
    """Resolved reliability artifacts for a Tier-3 host."""

    options: ReliabilityWiringOptions
    idempotency_store: IdempotencyStore | None
    circuit_breaker_config: IntegrationCircuitBreakerConfig


def wire_application_reliability(
    env: ApplicationEnvironmentProfile,
    *,
    idempotency_db_path: Path | None = None,
) -> ApplicationReliabilityWiring:
    """Materialize idempotency store and circuit breaker config from environment profile."""
    options = resolve_reliability_wiring_options(env.reliability_profile)
    idempotency_store: IdempotencyStore | None = None
    if options.idempotency_enabled:
        if idempotency_db_path is not None:
            idempotency_store = SQLiteIdempotencyStore(str(idempotency_db_path))
        else:
            idempotency_store = InMemoryIdempotencyStore()

    circuit_breaker_config = IntegrationCircuitBreakerConfig(
        failure_threshold=options.circuit_breaker_failure_threshold,
    )
    return ApplicationReliabilityWiring(
        options=options,
        idempotency_store=idempotency_store,
        circuit_breaker_config=circuit_breaker_config,
    )


def apply_reliability_task_defaults(task: Task, env: ApplicationEnvironmentProfile) -> Task:
    """Inject autonomy and resilience policy into task contract (REL-ADV)."""
    from intergrax.runtime.task.task import Task as _Task

    _ = _Task
    reliability = env.reliability_profile
    if task.options.governance.autonomy_level is None:
        task.options.governance.autonomy_level = reliability.default_autonomy_level
    task.metadata["autonomy_level"] = task.options.governance.autonomy_level.value
    task.metadata["resilience_policy.v1"] = reliability.resilience_policy.model_dump()
    task.metadata["max_interrupts_per_run"] = task.options.governance.max_interrupts_per_run
    task.metadata["autonomy_level_set"] = True
    replan_ctx = resolve_replan_policy_context(env)
    if replan_ctx:
        task.metadata["replan_policy.v1"] = replan_ctx
    compensation = resolve_compensation_flow(env)
    if compensation is not None:
        task.metadata["compensation_flow.v1"] = {
            "step_count": len(compensation.steps),
            "handler_ids": sorted(compensation.handlers.keys()),
        }
    task = apply_partial_results_task_defaults(task, env)
    task.sync_metadata()
    return task


def apply_reliability_governance_wiring(
    nexus: NexusLoop,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Attach autonomy governance middleware from reliability profile (REL-ADV.3)."""
    reliability = env.reliability_profile
    _attach_middleware(
        nexus,
        AutonomyGovernanceMiddleware(
            execution_mode=env.execution_mode,
            default_autonomy=reliability.default_autonomy_level,
            tenant_ceiling=reliability.tenant_autonomy_ceiling,
        ),
    )
