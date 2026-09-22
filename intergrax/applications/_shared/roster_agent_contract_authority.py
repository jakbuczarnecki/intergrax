# © Artur Czarnecki. All rights reserved.

"""Revision-bound and manifest-scoped AgentContract authority for STRICT consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from intergrax.agent_distribution.agent_contract_authority import (
    AgentPackageContractAuthorityService,
    PackageAgentContractAuthorityError,
)
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.agent_distribution.stores import AgentArtifactMetadataStore
from intergrax.applications._shared.wiring import (
    _index_manifest_bindings,
    binding_from_roster_entry,
)
from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract


class StrictAgentContractAuthorityError(ValueError):
    """STRICT production path lacks revision-bound AgentContract authority."""


@dataclass(frozen=True, slots=True)
class ManifestAgentContractAuthority:
    """Explicit per-contract snapshots for manifest-scoped STRICT gates."""

    contracts_by_id: Mapping[str, AgentContract]

    def contract_for_binding(self, binding: AgentBinding) -> AgentContract:
        contract_id = resolve_binding_contract_id(binding)
        contract = self.contracts_by_id.get(contract_id)
        if contract is None:
            raise StrictAgentContractAuthorityError(
                f"missing manifest contract authority for {contract_id!r}"
            )
        return contract


@dataclass(frozen=True, slots=True)
class RosterAgentContractAuthority:
    """Package-digest-bound contract authority for one frozen runtime revision."""

    _service: AgentPackageContractAuthorityService
    _package_digest_by_contract_id: Mapping[str, str]

    @classmethod
    def from_revision_roster(
        cls,
        *,
        artifact_metadata_store: AgentArtifactMetadataStore,
        runtime_revision: RuntimeRevision,
        effective_roster: EffectiveRoster,
        manifest: ApplicationManifest,
    ) -> RosterAgentContractAuthority:
        manifest_bindings = _index_manifest_bindings(manifest)
        trusted_digests = frozenset(runtime_revision.installed_agent_package_digests)
        digest_by_contract: dict[str, str] = {}
        for entry in effective_roster.entries:
            if not entry.effective_enablement:
                continue
            if entry.package_digest not in trusted_digests:
                raise StrictAgentContractAuthorityError(
                    f"roster entry {entry.logical_agent_id!r} package_digest not in revision"
                )
            binding = binding_from_roster_entry(entry, manifest_bindings)
            contract_id = resolve_binding_contract_id(binding)
            prior = digest_by_contract.get(contract_id)
            if prior is not None and prior != entry.package_digest:
                raise StrictAgentContractAuthorityError(
                    f"conflicting package digest for contract {contract_id!r}"
                )
            digest_by_contract[contract_id] = entry.package_digest
        return cls(
            _service=AgentPackageContractAuthorityService(artifact_metadata_store),
            _package_digest_by_contract_id=digest_by_contract,
        )

    def contract_for_binding(self, binding: AgentBinding) -> AgentContract:
        contract_id = resolve_binding_contract_id(binding)
        package_digest = self._package_digest_by_contract_id.get(contract_id)
        if package_digest is None:
            raise StrictAgentContractAuthorityError(
                f"contract {contract_id!r} not in revision-bound roster authority"
            )
        try:
            return self._service.resolve_contract(
                package_digest=package_digest,
                contract_id=contract_id,
            )
        except PackageAgentContractAuthorityError as exc:
            raise StrictAgentContractAuthorityError(str(exc)) from exc


ContractAuthority = ManifestAgentContractAuthority | RosterAgentContractAuthority


def resolve_roster_agent_contract(
    binding: AgentBinding,
    *,
    contract_authority: ContractAuthority | None,
    allow_compatibility_resolver: bool = False,
) -> AgentContract:
    """Resolve AgentContract for roster governance (STRICT uses authority only)."""
    if contract_authority is not None:
        return contract_authority.contract_for_binding(binding)
    if allow_compatibility_resolver:
        from intergrax.applications._shared.agent_resolution import (
            resolve_agent_contract_from_binding,
        )

        return resolve_agent_contract_from_binding(binding)
    raise StrictAgentContractAuthorityError(
        "STRICT agent contract resolution requires revision-bound authority"
    )


def materialize_manifest_contract_authority_lab_compat(
    manifest: ApplicationManifest,
) -> ManifestAgentContractAuthority:
    """Compatibility-only manifest authority for lab/monorepo gates (not production)."""
    from intergrax.applications._shared.agent_resolution import (
        resolve_agent_contract_from_binding,
    )

    contracts: dict[str, AgentContract] = {}
    for binding in manifest.enabled_agents():
        contract_id = resolve_binding_contract_id(binding)
        contracts[contract_id] = resolve_agent_contract_from_binding(binding)
    return ManifestAgentContractAuthority(contracts_by_id=contracts)


__all__ = [
    "ContractAuthority",
    "ManifestAgentContractAuthority",
    "RosterAgentContractAuthority",
    "StrictAgentContractAuthorityError",
    "materialize_manifest_contract_authority_lab_compat",
    "resolve_roster_agent_contract",
]
