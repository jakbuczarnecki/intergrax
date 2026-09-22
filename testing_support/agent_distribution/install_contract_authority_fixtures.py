# © Artur Czarnecki. All rights reserved.

"""Test/reference helpers for package AgentContract authority on install."""

from __future__ import annotations

import sys

from intergrax.agent_distribution.agent_contract_authority import (
    PackageAgentContractAuthorityRecord,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.applications._shared.roster_agent_contract_authority import (
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract


def reference_install_contract_authority_records(
    manifest: ApplicationManifest,
    *,
    package_digest: str,
    distribution_package_id: str,
    artifact_store_ref: str,
    agent_project_metadata_ref: str,
) -> tuple[PackageAgentContractAuthorityRecord, ...]:
    """Build digest-bound authority records for reference lifecycle installs (tests only)."""
    authority = materialize_manifest_contract_authority_lab_compat(manifest)
    records: list[PackageAgentContractAuthorityRecord] = []
    for binding in manifest.enabled_agents():
        contract = authority.contract_for_binding(binding)
        contract_id = resolve_binding_contract_id(binding)
        if contract.id != contract_id:
            contract = contract.model_copy(update={"id": contract_id})
        records.append(
            PackageAgentContractAuthorityRecord.from_validated_contract(
                package_digest=package_digest,
                distribution_package_id=distribution_package_id,
                artifact_store_ref=artifact_store_ref,
                agent_project_metadata_ref=agent_project_metadata_ref,
                contract=contract,
            )
        )
    return tuple(records)


def reference_echo_agent_contract() -> AgentContract:
    """Load the echo reference contract template (tests / lab proofs only)."""
    sys.path.insert(0, "agents/echo")
    try:
        from contract import build_agent_contract

        return build_agent_contract()
    finally:
        sys.path.pop(0)


def binding_agent_contract(
    *,
    contract_id: str,
    version: str,
    description: str,
) -> AgentContract:
    """Build a manifest-bound contract with the same logical id and distinct snapshots."""
    template = reference_echo_agent_contract()
    return template.model_copy(
        update={
            "id": contract_id,
            "version": version,
            "description": description,
        }
    )


def declared_contract_for(contract: AgentContract) -> AgentPackageContractDeclaration:
    return AgentPackageContractDeclaration(
        contract_id=contract.id,
        contract_version=contract.version,
        capabilities=tuple(
            item.id if hasattr(item, "id") else str(item)
            for item in contract.capabilities
        ),
        skill_ids=tuple(
            skill.skill_id for skill in contract.skills if skill.skill_id
        ),
        tool_ids=tuple(
            tool.tool_id
            for tool in getattr(contract, "allowed_tools", ())
            if getattr(tool, "tool_id", None)
        ),
    )


def package_contract_authority_record(
    *,
    contract: AgentContract,
    package_digest: str,
    distribution_package_id: str,
    artifact_store_ref: str,
    agent_project_metadata_ref: str,
) -> PackageAgentContractAuthorityRecord:
    return PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=package_digest,
        distribution_package_id=distribution_package_id,
        artifact_store_ref=artifact_store_ref,
        agent_project_metadata_ref=agent_project_metadata_ref,
        contract=contract,
    )
