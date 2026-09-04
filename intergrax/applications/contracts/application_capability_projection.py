# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Manifest → application capability descriptor projection (Stage 13)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    ApplicationCapabilityProjectionConflict,
)


def resolve_binding_contract_id(binding: AgentBinding) -> str:
    """Resolve enabled binding contract id declaratively (no agent instantiation)."""
    if binding.contract_id:
        return binding.contract_id
    raise ApplicationCapabilityProjectionConflict(
        "enabled AgentBinding lacks contract_id; capability projection requires declarative identity",
    )


def application_capability_descriptor_from_manifest(
    manifest: ApplicationManifest,
) -> ApplicationCapabilityDescriptor:
    """Project one application manifest into architecture discovery metadata."""
    contract_ids: list[str] = []
    for binding in manifest.enabled_agents():
        contract_ids.append(resolve_binding_contract_id(binding))
    return ApplicationCapabilityDescriptor(
        application_id=manifest.app_id,
        application_version=manifest.version,
        agent_contract_ids=tuple(contract_ids),
        default_capability=manifest.default_capability,
    )
