# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production composition for governed RuntimeToolInvoker (GR-10-R8)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from intergrax.contracts.canonical_inner_governance import CanonicalInnerExecutionGuardPort
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_canonical_inner_execution_guard,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.sandbox.isolation_gate import SandboxAvailabilityProvider
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.invocation_wiring import ToolInvocationWiringResolver
from intergrax.tools.registry.read import ToolRegistryRead

if TYPE_CHECKING:
    from intergrax.contracts.execution.crash_injection import (
        ToolRuntimeEffectCrashInjectionPort,
    )
    from intergrax.contracts.external_operation_cancellation import (
        ExternalOperationCancellationPort,
    )
    from intergrax.runtime.external_operations.external_operation_state_store import (
        ExternalOperationStateStore,
    )
    from intergrax.runtime.external_operations.external_operation_ownership import (
        ProcessLocalExternalOperationOwner,
    )
    from intergrax.tools.tool_executor import ToolExecutor


class ProductionRuntimeToolInvokerCompositionError(RuntimeError):
    """Fail closed when production orchestration tool invoker cannot be wired."""


def build_production_runtime_tool_invoker(
    *,
    registry: ToolRegistryRead,
    executor: ToolExecutor | None = None,
    inner_execution_guard: CanonicalInnerExecutionGuardPort | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None = None,
    scope_policy: ToolScopePolicy | None = None,
    pre_effect_coordinator: IdempotencyPreEffectCoordinator | None = None,
    idempotency_store: IdempotencyStore | None = None,
    sandbox_availability: SandboxAvailabilityProvider | None = None,
    agent_runtime_governance: AgentRuntimeGovernancePort | None = None,
    dependency_attempt_boundary: DependencyAttemptExecutionBoundary | None = None,
    external_operation_store: ExternalOperationStateStore | None = None,
    external_operation_owner: ProcessLocalExternalOperationOwner | None = None,
    external_operation_cancellation_port: ExternalOperationCancellationPort | None = None,
    invocation_wiring_resolver: ToolInvocationWiringResolver | None = None,
    production_mode: bool = False,
    effect_crash_injection: ToolRuntimeEffectCrashInjectionPort | None = None,
) -> RuntimeToolInvoker:
    """Wire canonical inner guard + production-required governance before tool effects."""
    if production_mode and agent_runtime_governance is None:
        raise ProductionRuntimeToolInvokerCompositionError(
            "agent_runtime_governance is required when production_mode=True",
        )
    guard = inner_execution_guard
    if production_mode:
        if guard is None:
            guard = build_default_canonical_inner_execution_guard()
        if guard is None:
            raise ProductionRuntimeToolInvokerCompositionError(
                "canonical inner execution guard is required when production_mode=True",
            )
    mse_authorization = meaningful_side_effect_authorization
    if production_mode and mse_authorization is None:
        raise ProductionRuntimeToolInvokerCompositionError(
            "meaningful_side_effect_authorization is required when production_mode=True",
        )
    resolved_executor = executor or RegistryToolExecutor(registry)
    coordinator = pre_effect_coordinator
    if coordinator is None and idempotency_store is not None:
        coordinator = IdempotencyPreEffectCoordinator(idempotency_store=idempotency_store)
    return RuntimeToolInvoker(
        registry=registry,
        executor=resolved_executor,
        scope_policy=scope_policy,
        pre_effect_coordinator=coordinator,
        sandbox_availability=sandbox_availability,
        agent_runtime_governance=agent_runtime_governance,
        inner_execution_guard=guard,
        meaningful_side_effect_authorization=mse_authorization,
        dependency_attempt_boundary=dependency_attempt_boundary,
        external_operation_store=external_operation_store,
        external_operation_owner=external_operation_owner,
        external_operation_cancellation_port=external_operation_cancellation_port,
        invocation_wiring_resolver=invocation_wiring_resolver,
        effect_crash_injection=effect_crash_injection,
    )


def recompose_runtime_tool_invoker_with_idempotency_store(
    invoker: RuntimeToolInvoker,
    *,
    idempotency_store: IdempotencyStore,
    production_mode: bool,
) -> RuntimeToolInvoker:
    """Delegate idempotency overlay to RuntimeToolInvoker owner-controlled reconfiguration."""
    return invoker.with_idempotency_store(
        idempotency_store,
        production_mode=production_mode,
    )


__all__ = [
    "ProductionRuntimeToolInvokerCompositionError",
    "build_production_runtime_tool_invoker",
    "recompose_runtime_tool_invoker_with_idempotency_store",
]
