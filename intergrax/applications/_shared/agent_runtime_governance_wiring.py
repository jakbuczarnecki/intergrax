# © Artur Czarnecki. All rights reserved.

"""Wire NPSC-4 agent runtime governance into production RuntimeConfig (U3 / EP-13)."""

from __future__ import annotations

from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)


class AgentRuntimeGovernanceMaterializationError(ValueError):
    """Fail-closed when production governance grants cannot be resolved declaratively."""


def _require_non_empty_tenant_id(tenant_id: str) -> str:
    normalized = tenant_id.strip()
    if not normalized:
        raise AgentRuntimeGovernanceMaterializationError(
            "tenant_id is required for agent runtime governance capability grants",
        )
    return normalized


def _capability_frozenset_from_contract(contract: AgentContract) -> frozenset[str]:
    capabilities = tuple(
        cap.strip() for cap in contract.capabilities if cap and str(cap).strip()
    )
    if not capabilities:
        raise AgentRuntimeGovernanceMaterializationError(
            f"registered AgentContract {contract.id!r} declares no capability ids",
        )
    return frozenset(capabilities)


def validated_capabilities_for_binding(
    binding: AgentBinding,
    agent_registry: AgentRegistryRead,
) -> frozenset[str]:
    """Resolve capability ids from registry contract metadata (no agent instantiation)."""
    agent_id = resolve_binding_contract_id(binding)
    if not agent_registry.has(agent_id):
        raise AgentRuntimeGovernanceMaterializationError(
            f"agent {agent_id!r} is not registered; "
            "capability grants require revision-bound or manifest registry contracts",
        )
    contract = agent_registry.get_contract(agent_id)
    contract_caps = _capability_frozenset_from_contract(contract)
    if binding.capabilities:
        declared = frozenset(binding.capabilities)
        if declared != contract_caps:
            raise AgentRuntimeGovernanceMaterializationError(
                f"manifest capabilities for {agent_id!r} do not match registered "
                f"AgentContract.capabilities",
            )
    return contract_caps


def capability_grants_from_application_manifest(
    manifest: ApplicationManifest,
    *,
    tenant_id: str,
    agent_registry: AgentRegistryRead,
) -> tuple[CapabilityGrant, ...]:
    """Materialize capability grants from enabled roster agents and registry contracts."""
    normalized_tenant = _require_non_empty_tenant_id(tenant_id)
    grants: list[CapabilityGrant] = []
    for binding in manifest.enabled_agents():
        agent_id = resolve_binding_contract_id(binding)
        allowed = validated_capabilities_for_binding(binding, agent_registry)
        grants.append(
            CapabilityGrant(
                agent_id=agent_id,
                tenant_id=normalized_tenant,
                allowed_capabilities=allowed,
            ),
        )
    if not grants:
        raise AgentRuntimeGovernanceMaterializationError(
            f"{manifest.app_id}: no enabled agents to materialize governance grants for",
        )
    return tuple(grants)


def apply_agent_runtime_governance_to_config(
    config: RuntimeConfig,
    *,
    capability_grants: tuple[CapabilityGrant, ...],
) -> None:
    """Attach mandatory agent runtime governance for production tool execution."""
    if not config.production_mode:
        return
    boundary = build_agent_runtime_governance_boundary(capability_grants=capability_grants)
    config.agent_runtime_governance = boundary


def resolve_agent_runtime_governance_port(
    config: RuntimeConfig,
) -> AgentRuntimeGovernancePort | None:
    """Typed accessor for RuntimeContext tool invoker wiring."""
    governance = config.agent_runtime_governance
    if governance is None:
        return None
    return governance
