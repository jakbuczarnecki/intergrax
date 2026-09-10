# © Artur Czarnecki. All rights reserved.

"""Wire NPSC-4 agent runtime governance into production RuntimeConfig (U3 / EP-13)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)


def capability_grants_from_application_manifest(
    manifest: ApplicationManifest,
    *,
    tenant_id: str,
) -> tuple[CapabilityGrant, ...]:
    """Materialize capability grants from enabled roster agents (contract capabilities)."""
    normalized_tenant = tenant_id.strip() or "default-tenant"
    grants: list[CapabilityGrant] = []
    for binding in manifest.enabled_agents():
        contract = binding.resolved_agent_type()().get_contract()
        capabilities = tuple(cap.strip() for cap in contract.capabilities if cap and str(cap).strip())
        if not capabilities:
            continue
        grants.append(
            CapabilityGrant(
                agent_id=contract.id,
                tenant_id=normalized_tenant,
                allowed_capabilities=frozenset(capabilities),
            ),
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
