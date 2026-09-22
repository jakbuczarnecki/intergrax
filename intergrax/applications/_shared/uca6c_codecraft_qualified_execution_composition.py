# © Artur Czarnecki. All rights reserved.

"""Production composition for UCA-6C qualified CodeCraft catalog tool execution.

Bootstrap owner for ``code.exec`` catalog registration: application/host calls
``bootstrap_uca6c_code_exec_catalog_tools`` once at startup (or relies on
``ensure_code_exec_registered`` during host-owned composition construction).
Registration must not occur per execution/resume.

Default ``StaticToolScopePolicy({CODE_EXEC_TOOL_ID})`` narrows tool authority to the
qualified CodeCraft execution surface only; host ``scope_policy`` may narrow further
but must not widen beyond the injected policy.
"""

from __future__ import annotations

from intergrax.applications._shared.agent_runtime_governance_wiring import (
    capability_grants_from_application_manifest,
)
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvoker,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_handler,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.suspended_operation.composition import (
    validate_suspended_operation_store_for_production,
    wire_default_suspended_operation_codec_registry,
    wire_suspended_execution_operation_store,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHostDependencies,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy, ToolScopePolicy
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.tools.providers.sandbox.bundle import (
    CODE_EXEC_TOOL_ID,
    register_sandbox_tools,
)


class Uca6cCodecraftQualifiedExecutionCompositionError(RuntimeError):
    """Fail closed when production CodeCraft tool invocation cannot be wired."""


def bootstrap_uca6c_code_exec_catalog_tools(tool_wiring: ApplicationToolWiring) -> None:
    """Register sandbox catalog tools (including ``code.exec``) on the application registry."""
    registry = tool_wiring.registry
    if not registry.has(CODE_EXEC_TOOL_ID):
        register_sandbox_tools(registry, tool_wiring.wiring_context)


def _internal_hitl_continuation(
    continuation_dependencies: ExecutionEngineContinuationDependencies,
) -> InternalOrchestrationContinuation:
    return InternalOrchestrationContinuation(
        port=continuation_dependencies.continuation,
        lifecycle_driver=continuation_dependencies.lifecycle_driver,
        projection_sink=None,
    )


def build_continuation_aware_catalog_tool_host_dependencies(
    *,
    document_store: ConditionalDocumentStore | None,
    continuation_dependencies: ExecutionEngineContinuationDependencies,
    production_mode: bool,
) -> ContinuationAwareCatalogToolHostDependencies:
    store = wire_suspended_execution_operation_store(document_store=document_store)
    if production_mode:
        validate_suspended_operation_store_for_production(store)
    return ContinuationAwareCatalogToolHostDependencies(
        suspended_operation_store=store,
        hitl_continuation=_internal_hitl_continuation(continuation_dependencies),
        codec_registry=wire_default_suspended_operation_codec_registry(),
    )


def build_execution_bound_catalog_tool_invoker_for_qualified_capability(
    tool_wiring: ApplicationToolWiring,
    environment: ApplicationEnvironmentProfile,
    *,
    caller_agent_id: str,
    tenant_id: str,
    manifest: ApplicationManifest | None = None,
    agent_registry: AgentRegistryRead | None = None,
    scope_policy: ToolScopePolicy | None = None,
    canonical_inner_execution_guard: CanonicalInnerExecutionGuardPort | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort
    | None = None,
    ensure_code_exec_registered: bool = True,
    document_store: ConditionalDocumentStore | None = None,
    continuation_dependencies: ExecutionEngineContinuationDependencies | None = None,
) -> ExecutionBoundCatalogToolInvoker:
    """Host-owned canonical ToolRuntime + trusted runtime state for qualified execution."""
    if not caller_agent_id.strip():
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "caller_agent_id is required for qualified capability catalog invocation",
        )
    if not tenant_id.strip():
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "tenant_id is required for qualified capability catalog invocation",
        )
    registry = tool_wiring.registry
    if ensure_code_exec_registered:
        bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    production_mode = environment.execution_mode.value == "strict"
    if production_mode and document_store is None:
        document_store = InMemoryDocumentStore()
    governance = None
    if production_mode:
        if manifest is None or agent_registry is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires manifest and agent_registry for tool governance",
            )
        if meaningful_side_effect_authorization is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires meaningful_side_effect_authorization "
                "for tool governance",
            )
        grants = capability_grants_from_application_manifest(
            manifest,
            tenant_id=tenant_id,
            agent_registry=agent_registry,
        )
        governance = build_agent_runtime_governance_boundary(capability_grants=grants)
    resolved_scope = scope_policy or StaticToolScopePolicy(
        allowed_tools={CODE_EXEC_TOOL_ID},
    )
    invoker = build_production_runtime_tool_invoker(
        registry=registry,
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        agent_runtime_governance=governance,
        inner_execution_guard=canonical_inner_execution_guard,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        scope_policy=resolved_scope,
        production_mode=production_mode,
    )
    policy_bundle = wire_policy_bundle(environment)
    continuation_aware_dependencies = None
    if production_mode:
        if continuation_dependencies is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires execution continuation dependencies "
                "for canonical HITL re-entry",
            )
        continuation_aware_dependencies = (
            build_continuation_aware_catalog_tool_host_dependencies(
                document_store=document_store,
                continuation_dependencies=continuation_dependencies,
                production_mode=True,
            )
        )
    elif continuation_dependencies is not None:
        continuation_aware_dependencies = (
            build_continuation_aware_catalog_tool_host_dependencies(
                document_store=document_store,
                continuation_dependencies=continuation_dependencies,
                production_mode=False,
            )
        )
    return NexusExecutionBoundCatalogToolInvoker(
        tool_invoker=invoker,
        policy_bundle=policy_bundle,
        caller_agent_id=caller_agent_id.strip(),
        production_mode=production_mode,
        continuation_aware_dependencies=continuation_aware_dependencies,
    )


def build_production_codecraft_qualified_capability_execution_handler(
    tool_wiring: ApplicationToolWiring,
    environment: ApplicationEnvironmentProfile,
    *,
    caller_agent_id: str,
    tenant_id: str,
    manifest: ApplicationManifest | None = None,
    agent_registry: AgentRegistryRead | None = None,
    scope_policy: ToolScopePolicy | None = None,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort
    | None = None,
    canonical_inner_execution_guard: CanonicalInnerExecutionGuardPort | None = None,
    ensure_code_exec_registered: bool = True,
    side_effect_recorder: list[str] | None = None,
    document_store: ConditionalDocumentStore | None = None,
    continuation_dependencies: ExecutionEngineContinuationDependencies | None = None,
) -> CodeCraftQualifiedCapabilityExecutionHandler:
    """Production stack: host ToolRuntime → execution-bound invoker → CodeCraft handler."""
    if (
        continuation_dependencies is None
        and environment.execution_mode.value == "strict"
    ):
        continuation_dependencies = wire_execution_engine_continuation_dependencies()
    if environment.execution_mode.value == "strict" and document_store is None:
        document_store = InMemoryDocumentStore()
    catalog_invoker = (
        build_execution_bound_catalog_tool_invoker_for_qualified_capability(
            tool_wiring,
            environment,
            caller_agent_id=caller_agent_id,
            tenant_id=tenant_id,
            manifest=manifest,
            agent_registry=agent_registry,
            scope_policy=scope_policy,
            meaningful_side_effect_authorization=meaningful_side_effect_authorization,
            canonical_inner_execution_guard=canonical_inner_execution_guard,
            ensure_code_exec_registered=ensure_code_exec_registered,
            document_store=document_store,
            continuation_dependencies=continuation_dependencies,
        )
    )
    return build_codecraft_qualified_capability_execution_handler(
        tool_wiring.wiring_context,
        catalog_tool_invoker=catalog_invoker,
        side_effect_recorder=side_effect_recorder,
    )


__all__ = [
    "Uca6cCodecraftQualifiedExecutionCompositionError",
    "bootstrap_uca6c_code_exec_catalog_tools",
    "build_continuation_aware_catalog_tool_host_dependencies",
    "build_execution_bound_catalog_tool_invoker_for_qualified_capability",
    "build_production_codecraft_qualified_capability_execution_handler",
]
