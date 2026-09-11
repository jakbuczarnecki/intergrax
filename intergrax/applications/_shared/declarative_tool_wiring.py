# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for ACP declarative catalog tool invoker."""

from __future__ import annotations

from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
)
from intergrax.applications._shared.agent_runtime_governance_wiring import (
    AgentRuntimeGovernanceMaterializationError,
    capability_grants_from_application_manifest,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)


def build_declarative_invoker_from_tool_wiring(
    tool_wiring: ApplicationToolWiring,
    *,
    idempotency_store: IdempotencyStore | None = None,
    agent_runtime_governance: AgentRuntimeGovernancePort | None = None,
    production_mode: bool = False,
) -> CatalogDeclarativeToolInvoker | None:
    """Materialize catalog invoker when host tool profile enables catalog tools."""
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        return None
    if production_mode and agent_runtime_governance is None:
        raise AgentRuntimeGovernanceMaterializationError(
            "production declarative tool invoker requires agent_runtime_governance",
        )
    coordinator = (
        IdempotencyPreEffectCoordinator(idempotency_store=idempotency_store)
        if idempotency_store is not None
        else None
    )
    invoker = RuntimeToolInvoker(
        registry=tool_wiring.registry,
        executor=RegistryToolExecutor(tool_wiring.registry),
        pre_effect_coordinator=coordinator,
        sandbox_availability=sandbox_availability_provider(tool_wiring.wiring_context),
        agent_runtime_governance=agent_runtime_governance,
    )
    return CatalogDeclarativeToolInvoker(
        tool_invoker=invoker,
        production_mode=production_mode,
    )


def build_declarative_invoker_for_application_host(
    tool_wiring: ApplicationToolWiring,
    environment: ApplicationEnvironmentProfile,
    *,
    manifest: ApplicationManifest,
    agent_registry: AgentRegistryRead,
    tenant_id: str,
    idempotency_store: IdempotencyStore | None = None,
) -> CatalogDeclarativeToolInvoker | None:
    """Compose declarative invoker with strict-mode agent governance (U5 / EP-14)."""
    production_mode = environment.execution_mode.value == "strict"
    governance: AgentRuntimeGovernancePort | None = None
    if production_mode:
        grants = capability_grants_from_application_manifest(
            manifest,
            tenant_id=tenant_id,
            agent_registry=agent_registry,
        )
        governance = build_agent_runtime_governance_boundary(capability_grants=grants)
    return build_declarative_invoker_from_tool_wiring(
        tool_wiring,
        idempotency_store=idempotency_store,
        agent_runtime_governance=governance,
        production_mode=production_mode,
    )
