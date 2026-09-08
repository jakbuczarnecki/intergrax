# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace catalog source tests."""

from __future__ import annotations

import pytest

from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceIdentity, CapabilitySourceKind
from intergrax.contracts.marketplace import CommercialModel, MarketplaceCommercialMetadata, MarketplacePublisherMetadata
from intergrax.marketplace import MarketplaceCapabilityCatalogSource, MarketplaceCatalogConfigurationError, MarketplaceListingRecord

pytestmark = pytest.mark.unit


def _official_source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="official.intergrax.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _enterprise_source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="enterprise.acme.marketplace",
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )


def _third_party_source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="partner.widgets.marketplace",
        source_kind=CapabilitySourceKind.THIRD_PARTY,
    )


def test_official_marketplace_source_reads_entries_deterministically() -> None:
    source = MarketplaceCapabilityCatalogSource(
        source=_official_source(),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.marketplace.search",
                version_label="2.0.0",
                publisher="intergrax",
                publisher_metadata=MarketplacePublisherMetadata(
                    publisher_id="intergrax",
                    display_name="Intergrax",
                ),
                commercial_metadata=MarketplaceCommercialMetadata(
                    commercial_model=CommercialModel.FREE,
                ),
            ),
            MarketplaceListingRecord(
                kind=CapabilityKind.AGENT,
                logical_id="agents.research.v1",
                publisher="intergrax",
            ),
        ),
    )
    first = source.read_entries()
    second = source.read_entries()
    assert first == second
    assert [entry.identity.logical.logical_id for entry in first] == [
        "agents.research.v1",
        "tools.marketplace.search",
    ]
    assert all(
        entry.identity.source.source_kind is CapabilitySourceKind.OFFICIAL for entry in first
    )


def test_private_marketplace_source_preserves_source_kind() -> None:
    source = MarketplaceCapabilityCatalogSource(
        source=_enterprise_source(),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.SKILL,
                logical_id="skills.enterprise.pack",
                publisher="acme",
                commercial_metadata=MarketplaceCommercialMetadata(
                    commercial_model=CommercialModel.INTERNAL,
                ),
            ),
        ),
    )
    entry = source.read_entries()[0]
    assert entry.identity.source.source_kind is CapabilitySourceKind.ENTERPRISE_PRIVATE
    listing = source.read_listings()[0]
    assert listing.commercial_metadata is not None
    assert listing.commercial_metadata.commercial_model is CommercialModel.INTERNAL


def test_third_party_marketplace_source_supported() -> None:
    source = MarketplaceCapabilityCatalogSource(
        source=_third_party_source(),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.partner.widget",
                publisher="widgets-inc",
            ),
        ),
    )
    assert source.read_entries()[0].identity.source.source_kind is CapabilitySourceKind.THIRD_PARTY


def test_marketplace_source_rejects_builtin_kind() -> None:
    with pytest.raises(MarketplaceCatalogConfigurationError, match="marketplace catalog source"):
        MarketplaceCapabilityCatalogSource(
            source=CapabilitySourceIdentity(
                source_id="builtin.marketplace",
                source_kind=CapabilitySourceKind.BUILTIN,
            ),
            records=(
                MarketplaceListingRecord(
                    kind=CapabilityKind.TOOL,
                    logical_id="tools.bad",
                ),
            ),
        )


def test_marketplace_snapshot_rejects_duplicate_identity() -> None:
    record = MarketplaceListingRecord(
        kind=CapabilityKind.TOOL,
        logical_id="tools.duplicate",
    )
    with pytest.raises(MarketplaceCatalogConfigurationError, match="duplicate marketplace listing"):
        MarketplaceCapabilityCatalogSource(
            source=_official_source(),
            records=(record, record),
        )
