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
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_handler,
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
    if ensure_code_exec_registered and not registry.has(CODE_EXEC_TOOL_ID):
        register_sandbox_tools(registry, tool_wiring.wiring_context)
    production_mode = environment.execution_mode.value == "strict"
    governance = None
    if production_mode:
        if manifest is None or agent_registry is None:
            raise Uca6cCodecraftQualifiedExecutionCompositionError(
                "STRICT execution requires manifest and agent_registry for tool governance",
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
    return NexusExecutionBoundCatalogToolInvoker(
        tool_invoker=invoker,
        policy_bundle=policy_bundle,
        caller_agent_id=caller_agent_id.strip(),
        production_mode=production_mode,
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
    side_effect_recorder: list[str] | None = None,
) -> CodeCraftQualifiedCapabilityExecutionHandler:
    """Production stack: host ToolRuntime → execution-bound invoker → CodeCraft handler."""
    catalog_invoker = (
        build_execution_bound_catalog_tool_invoker_for_qualified_capability(
            tool_wiring,
            environment,
            caller_agent_id=caller_agent_id,
            tenant_id=tenant_id,
            manifest=manifest,
            agent_registry=agent_registry,
            scope_policy=scope_policy,
        )
    )
    return build_codecraft_qualified_capability_execution_handler(
        tool_wiring.wiring_context,
        catalog_tool_invoker=catalog_invoker,
        side_effect_recorder=side_effect_recorder,
    )


__all__ = [
    "Uca6cCodecraftQualifiedExecutionCompositionError",
    "build_execution_bound_catalog_tool_invoker_for_qualified_capability",
    "build_production_codecraft_qualified_capability_execution_handler",
]
