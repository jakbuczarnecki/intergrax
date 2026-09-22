# © Artur Czarnecki. All rights reserved.

"""Production composition for UCA-6C qualified CodeCraft catalog tool execution."""

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
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_handler,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.execution.suspended_operation.composition import (
    validate_document_store_for_production_suspended_operations,
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
    registry = tool_wiring.registry
    if not registry.has(CODE_EXEC_TOOL_ID):
        register_sandbox_tools(registry, tool_wiring.wiring_context)


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
    if not caller_agent_id.strip():
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "caller_agent_id is required for qualified capability catalog invocation",
        )
    if not tenant_id.strip():
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "tenant_id is required for qualified capability catalog invocation",
        )
    if ensure_code_exec_registered:
        bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    production_mode = environment.execution_mode.value == "strict"
    if production_mode and document_store is None:
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "STRICT execution requires explicit durable document_store",
        )
    if production_mode:
        validate_document_store_for_production_suspended_operations(document_store)
    governance = None
    if production_mode:
        if manifest is None or agent_registry is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires manifest and agent_registry for tool governance",
            )
        if meaningful_side_effect_authorization is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires meaningful_side_effect_authorization",
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
    if production_mode and continuation_dependencies is None:
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "STRICT execution requires execution continuation dependencies",
        )
    composition = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=wire_policy_bundle(environment),
        caller_agent_id=caller_agent_id,
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        production_mode=production_mode,
        scope_policy=resolved_scope,
        agent_runtime_governance=governance,
        canonical_inner_execution_guard=canonical_inner_execution_guard,
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        document_store=document_store,
        continuation_dependencies=continuation_dependencies,
        reentry_claim_owner_id=f"uca6c:{caller_agent_id.strip()}",
    )
    return composition.invoker


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
    if (
        continuation_dependencies is None
        and environment.execution_mode.value == "strict"
    ):
        raise Uca6cCodecraftQualifiedExecutionCompositionError(
            "STRICT execution requires injected execution continuation dependencies",
        )
    catalog_invoker = build_execution_bound_catalog_tool_invoker_for_qualified_capability(
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
    return build_codecraft_qualified_capability_execution_handler(
        tool_wiring.wiring_context,
        catalog_tool_invoker=catalog_invoker,
        side_effect_recorder=side_effect_recorder,
    )


__all__ = [
    "Uca6cCodecraftQualifiedExecutionCompositionError",
    "bootstrap_uca6c_code_exec_catalog_tools",
    "build_execution_bound_catalog_tool_invoker_for_qualified_capability",
    "build_production_codecraft_qualified_capability_execution_handler",
]
