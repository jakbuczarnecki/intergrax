# © Artur Czarnecki. All rights reserved.

"""EE composition seam for execution-bound catalog tool invocation (UCA-6C-R6-R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
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
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.sandbox.isolation_gate import SandboxAvailabilityProvider
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.registry.runtime import ToolRegistry


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
) -> ExecutionBoundCatalogToolComposition:
    tool_invoker = build_production_runtime_tool_invoker(
        registry=registry,
        sandbox_availability=sandbox_availability,
        agent_runtime_governance=agent_runtime_governance,
        inner_execution_guard=canonical_inner_execution_guard,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        scope_policy=scope_policy,
        production_mode=production_mode,
    )
    suspended_store = wire_suspended_execution_operation_store(
        document_store=document_store,
    )
    reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None = None
    continuation_aware_dependencies: ContinuationAwareCatalogToolHostDependencies | None = (
        None
    )
    if continuation_dependencies is not None:
        if production_mode:
            validate_production_suspended_operation_wiring(
                document_store=document_store,
                store=suspended_store,
            )
        continuation_aware_dependencies = ContinuationAwareCatalogToolHostDependencies(
            suspended_operation_store=suspended_store,
            hitl_continuation=InternalOrchestrationContinuation(
                port=continuation_dependencies.continuation,
                lifecycle_driver=continuation_dependencies.lifecycle_driver,
            ),
            codec_registry=wire_default_suspended_operation_codec_registry(),
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
    if continuation_dependencies is not None and continuation_aware_dependencies is not None:
        host = ContinuationAwareCatalogToolHost(
            tool_invoker=tool_invoker,
            dependencies=continuation_aware_dependencies,
        )
        reentry_coordinator = wire_execution_suspended_work_reentry_coordinator(
            store=suspended_store,
            continuation_port=continuation_dependencies.continuation,
            catalog_invoker=bound,
            catalog_host=host,
            claim_owner_id=reentry_claim_owner_id,
        )
        continuation_aware_dependencies = ContinuationAwareCatalogToolHostDependencies(
            suspended_operation_store=suspended_store,
            hitl_continuation=InternalOrchestrationContinuation(
                port=continuation_dependencies.continuation,
                lifecycle_driver=continuation_dependencies.lifecycle_driver,
                suspended_work_reentry_coordinator=reentry_coordinator,
            ),
            codec_registry=wire_default_suspended_operation_codec_registry(),
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
