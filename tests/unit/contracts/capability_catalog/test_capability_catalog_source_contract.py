# © Artur Czarnecki. All rights reserved.

"""CapabilityCatalogSource public contract smoke."""

from __future__ import annotations

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityCatalogSource,
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

pytestmark = pytest.mark.unit


class _ExternalCatalogSource:
    @property
    def source_id(self) -> str:
        return "external.catalog"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        source = CapabilitySourceIdentity(
            source_id=self.source_id,
            source_kind=CapabilitySourceKind.LOCAL,
        )
        return (
            CapabilityCatalogEntry(
                identity=CapabilityDiscoveryIdentity(
                    kind=CapabilityKind.AGENT,
                    source=source,
                    logical=CapabilityLogicalIdentity(
                        kind=CapabilityKind.AGENT,
                        logical_id="agents.external",
                    ),
                ),
                provenance=CapabilityProvenance(source=source),
            ),
        )


def test_external_source_structurally_satisfies_capability_catalog_source() -> None:
    source: CapabilityCatalogSource = _ExternalCatalogSource()
    assert source.read_entries()[0].identity.logical.logical_id == "agents.external"
