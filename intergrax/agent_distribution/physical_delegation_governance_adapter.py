# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""NPSC caller adapter: delegated subtask selection facts → governance request."""

from __future__ import annotations

from intergrax.agent_distribution.capability_matching import AgentCapabilityRequirement
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationCapabilityRequirement,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationSelectedIdentity,
)


def project_physical_delegation_selected_identity(
    selected_identity: AgentDiscoveryCandidateIdentity,
) -> PhysicalDelegationSelectedIdentity:
    return PhysicalDelegationSelectedIdentity(
        catalog_source_id=selected_identity.source.catalog_source_id,
        provider_kind=selected_identity.source.provider_kind.value,
        distribution_package_id=selected_identity.package.distribution_package_id,
        package_version=selected_identity.package.package_version,
        package_digest=selected_identity.package.package_digest,
    )


def project_physical_delegation_capability_requirement(
    capability_requirement: AgentCapabilityRequirement,
) -> PhysicalDelegationCapabilityRequirement:
    return PhysicalDelegationCapabilityRequirement(
        required_capability_ids=tuple(
            sorted(str(capability_id) for capability_id in capability_requirement.required_capability_ids),
        ),
    )


def build_physical_delegation_governance_request(
    *,
    delegation_id: str,
    task_scope_id: str,
    application_id: str,
    application_environment_id: str,
    capability_requirement: AgentCapabilityRequirement,
    selected_identity: AgentDiscoveryCandidateIdentity,
    principal: RequestIdentity,
    requested_permission_scopes: tuple[str, ...] | None = None,
) -> PhysicalDelegationGovernanceRequest:
    """Project post-selection delegated subtask facts into governance-owned request."""
    return PhysicalDelegationGovernanceRequest(
        delegation_id=delegation_id,
        task_scope_id=task_scope_id,
        application_id=application_id,
        application_environment_id=application_environment_id,
        principal=principal,
        capability_requirement=project_physical_delegation_capability_requirement(
            capability_requirement,
        ),
        selected_identity=project_physical_delegation_selected_identity(
            selected_identity,
        ),
        requested_permission_scopes=requested_permission_scopes or (),
    )


__all__ = [
    "build_physical_delegation_governance_request",
    "project_physical_delegation_capability_requirement",
    "project_physical_delegation_selected_identity",
]
