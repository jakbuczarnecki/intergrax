# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool bundle catalog → capability catalog entry adapter (Stage 2)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog.identity import (
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.provenance import CapabilityProvenance
from intergrax.tools.registry.catalog import ToolBundleEntry, iter_bundles

TOOL_BUILTIN_CATALOG_SOURCE_ID: Final = "tools.catalog.builtin"

_BUILTIN_SOURCE = CapabilitySourceIdentity(
    source_id=TOOL_BUILTIN_CATALOG_SOURCE_ID,
    source_kind=CapabilitySourceKind.BUILTIN,
)


def project_tool_bundle_entry(
    bundle: ToolBundleEntry,
    tool_id: str,
) -> CapabilityCatalogEntry:
    """Map one tool capability from a bundle row to a federated catalog entry."""
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_BUILTIN_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=tool_id,
            ),
        ),
        provenance=CapabilityProvenance(
            source=_BUILTIN_SOURCE,
            package_reference=bundle.bundle_id,
        ),
        display_label=tool_id,
    )


class ToolBundleCatalogSource:
    """Read-only adapter over the in-memory tool bundle catalog."""

    @property
    def source_id(self) -> str:
        return TOOL_BUILTIN_CATALOG_SOURCE_ID

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        entries: list[CapabilityCatalogEntry] = []
        for bundle in iter_bundles():
            for tool_id in bundle.tool_ids:
                entries.append(project_tool_bundle_entry(bundle, tool_id))
        return tuple(entries)
