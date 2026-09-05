# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent catalog → capability catalog entry adapter (Stage 2)."""

from __future__ import annotations

from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
    CatalogSourceProvider,
)
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance

_PROVIDER_KIND_TO_SOURCE_KIND: dict[CatalogProviderKind, CapabilitySourceKind] = {
    CatalogProviderKind.BUILTIN: CapabilitySourceKind.BUILTIN,
    CatalogProviderKind.LOCAL_DEVELOPER: CapabilitySourceKind.LOCAL,
    CatalogProviderKind.ENTERPRISE_PRIVATE: CapabilitySourceKind.ENTERPRISE_PRIVATE,
    CatalogProviderKind.OFFICIAL_CATALOG: CapabilitySourceKind.OFFICIAL,
    CatalogProviderKind.GOVERNED_THIRD_PARTY: CapabilitySourceKind.THIRD_PARTY,
}


def _map_source_identity(catalog_source: CatalogSourceIdentity) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=catalog_source.catalog_source_id,
        source_kind=_PROVIDER_KIND_TO_SOURCE_KIND[catalog_source.provider_kind],
    )


def project_agent_catalog_entry(entry: AgentCatalogEntry) -> CapabilityCatalogEntry:
    """Map one agent catalog row to a federated capability catalog entry."""
    source = _map_source_identity(entry.catalog_source)
    version_label = None
    if entry.version_channel_refs:
        version_label = entry.version_channel_refs[0].version_label
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.AGENT,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.AGENT,
                logical_id=entry.catalog_entry_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label=version_label,
            package_reference=entry.package_id_line,
            publisher=entry.publisher,
        ),
        display_label=entry.display_name,
    )


class AgentCatalogCapabilitySource:
    """Read-only adapter over ``CatalogSourceProvider``."""

    def __init__(self, provider: CatalogSourceProvider) -> None:
        self._provider = provider

    @property
    def source_id(self) -> str:
        return self._provider.catalog_source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return tuple(
            project_agent_catalog_entry(entry)
            for entry in self._provider.list_entries()
        )
