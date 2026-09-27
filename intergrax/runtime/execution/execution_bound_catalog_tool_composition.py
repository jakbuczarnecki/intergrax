# © Artur Czarnecki. All rights reserved.

"""EE composition seam for execution-bound catalog tool invocation (UCA-6C-R6-R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionConfiguration,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.runtime.resilience.dependency_attempt_boundary_composition import (
    materialize_tool_dependency_attempt_boundary,
)
from intergrax.contracts.execution_deadline.clock import UtcClockPort
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.execution.deadline_authority.system_clocks import SystemUtcClock
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
)
from intergrax.runtime.execution.suspended_operation.composition import (
    validate_production_suspended_operation_wiring,
    wire_default_suspended_operation_codec_registry,
    wire_execution_suspended_work_reentry_coordinator,
    wire_suspended_execution_operation_store,
)
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    ExecutionSuspendedWorkReentryCoordinator,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHost,
    ContinuationAwareCatalogToolHostDependencies,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.sandbox.isolation_gate import SandboxAvailabilityProvider
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.durable_invocation_wiring_binding_resolver import (
    DurableToolInvocationWiringBindingResolver,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from intergrax.contracts.execution.crash_injection import (
        ExecutionSuspendedWorkReentryCrashInjectionPort,
        ToolRuntimeEffectCrashInjectionPort,
    )
    from intergrax.contracts.idempotency_store import IdempotencyStore


@dataclass(frozen=True, slots=True)
class ExecutionBoundCatalogToolComposition:
    invoker: ExecutionBoundCatalogToolInvoker
    suspended_work_reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None


def build_execution_bound_catalog_tool_composition(
    *,
    registry: ToolRegistry,
    policy_bundle: RuntimePolicyBundle,
    caller_agent_id: str,
    sandbox_availability: SandboxAvailabilityProvider,
    production_mode: bool,
    scope_policy: ToolScopePolicy,
    agent_runtime_governance: AgentRuntimeGovernanceBoundary | None,
    canonical_inner_execution_guard: CanonicalInnerExecutionGuardPort | None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
    document_store: ConditionalDocumentStore | None,
    continuation_dependencies: ExecutionEngineContinuationDependencies | None,
    reentry_claim_owner_id: str,
    durable_wiring_binding_resolver: DurableToolInvocationWiringBindingResolver
    | None = None,
    task_checkpoint_store: TaskCheckpointPersistence | None = None,
    idempotency_store: IdempotencyStore | None = None,
    tool_executor: ToolExecutor | None = None,
    terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore | None = None,
    utc_clock: UtcClockPort | None = None,
    reentry_crash_injection: ExecutionSuspendedWorkReentryCrashInjectionPort | None = None,
    tool_runtime_effect_crash_injection: ToolRuntimeEffectCrashInjectionPort | None = None,
    dependency_concurrency_admission: DependencyConcurrencyAdmissionConfiguration | None = None,
) -> ExecutionBoundCatalogToolComposition:
    dependency_boundary = materialize_tool_dependency_attempt_boundary(
        dependency_concurrency_admission,
        production_mode=production_mode,
    )
    pre_effect_coordinator = (
        IdempotencyPreEffectCoordinator(idempotency_store=idempotency_store)
        if idempotency_store is not None
        else None
    )
    tool_invoker = build_production_runtime_tool_invoker(
        registry=registry,
        executor=tool_executor,
        sandbox_availability=sandbox_availability,
        agent_runtime_governance=agent_runtime_governance,
        inner_execution_guard=canonical_inner_execution_guard,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        scope_policy=scope_policy,
        idempotency_store=idempotency_store,
        pre_effect_coordinator=pre_effect_coordinator,
        production_mode=production_mode,
        effect_crash_injection=tool_runtime_effect_crash_injection,
        dependency_attempt_boundary=dependency_boundary,
    )
    shared_utc_clock = utc_clock if utc_clock is not None else SystemUtcClock()
    suspended_store = wire_suspended_execution_operation_store(
        document_store=document_store,
        utc_clock=shared_utc_clock,
    )
    reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None = None
    continuation_aware_dependencies: (
        ContinuationAwareCatalogToolHostDependencies | None
    ) = None
    if continuation_dependencies is not None:
        if production_mode:
            validate_production_suspended_operation_wiring(
                document_store=document_store,
                store=suspended_store,
            )
            if task_checkpoint_store is None:
                raise RuntimeError(
                    "STRICT execution requires task_checkpoint_store for "
                    "agent governance pause projection",
                )
        continuation_aware_dependencies = ContinuationAwareCatalogToolHostDependencies(
            suspended_operation_store=suspended_store,
            hitl_continuation=InternalOrchestrationContinuation(
                port=continuation_dependencies.continuation,
                lifecycle_driver=continuation_dependencies.lifecycle_driver,
            ),
            codec_registry=wire_default_suspended_operation_codec_registry(),
            task_checkpoint_store=task_checkpoint_store,
        )
    elif production_mode:
        raise RuntimeError(
            "STRICT execution requires execution continuation dependencies",
        )
    bound = NexusExecutionBoundCatalogToolInvoker(
        tool_invoker=tool_invoker,
        policy_bundle=policy_bundle,
        caller_agent_id=caller_agent_id.strip(),
        production_mode=production_mode,
        continuation_aware_dependencies=continuation_aware_dependencies,
    )
    if (
        continuation_dependencies is not None
        and continuation_aware_dependencies is not None
    ):
        if durable_wiring_binding_resolver is None:
            raise RuntimeError(
                "continuation-aware catalog invocation requires "
                "durable_wiring_binding_resolver",
            )
        host = ContinuationAwareCatalogToolHost(
            tool_invoker=tool_invoker,
            dependencies=continuation_aware_dependencies,
        )
        reentry_coordinator = wire_execution_suspended_work_reentry_coordinator(
            store=suspended_store,
            continuation_port=continuation_dependencies.continuation,
            tool_registry=registry,
            catalog_invoker=bound,
            catalog_host=host,
            binding_resolver=durable_wiring_binding_resolver,
            claim_owner_id=reentry_claim_owner_id,
            task_checkpoint_store=task_checkpoint_store,
            terminal_outcome_store=terminal_outcome_store,
            utc_clock=shared_utc_clock,
            crash_injection=reentry_crash_injection,
            pre_effect_coordinator=pre_effect_coordinator,
        )
        continuation_aware_dependencies = ContinuationAwareCatalogToolHostDependencies(
            suspended_operation_store=suspended_store,
            hitl_continuation=InternalOrchestrationContinuation(
                port=continuation_dependencies.continuation,
                lifecycle_driver=continuation_dependencies.lifecycle_driver,
                suspended_work_reentry_coordinator=reentry_coordinator,
            ),
            codec_registry=wire_default_suspended_operation_codec_registry(),
            task_checkpoint_store=task_checkpoint_store,
        )
        bound = NexusExecutionBoundCatalogToolInvoker(
            tool_invoker=tool_invoker,
            policy_bundle=policy_bundle,
            caller_agent_id=caller_agent_id.strip(),
            production_mode=production_mode,
            continuation_aware_dependencies=continuation_aware_dependencies,
        )
    return ExecutionBoundCatalogToolComposition(
        invoker=bound,
        suspended_work_reentry_coordinator=reentry_coordinator,
    )


__all__ = [
    "ExecutionBoundCatalogToolComposition",
    "build_execution_bound_catalog_tool_composition",
]
