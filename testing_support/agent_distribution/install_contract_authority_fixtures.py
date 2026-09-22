# © Artur Czarnecki. All rights reserved.

"""Test/reference helpers for package AgentContract authority on install."""

from __future__ import annotations

from intergrax.agent_distribution.agent_contract_authority import (
    PackageAgentContractAuthorityRecord,
)
from intergrax.applications._shared.roster_agent_contract_authority import (
    materialize_manifest_contract_authority_lab_compat,
)
from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.applications.contracts.manifest import ApplicationManifest


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
