# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace product contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    CommercialModel,
    MarketplaceCommercialMetadata,
    MarketplacePublisherMetadata,
)
from intergrax.marketplace.listing import MarketplaceCapabilityListing

pytestmark = pytest.mark.unit


def _entry(
    *,
    publisher: str | None = "acme",
    source_kind: CapabilitySourceKind = CapabilitySourceKind.OFFICIAL,
) -> CapabilityCatalogEntry:
    source = CapabilitySourceIdentity(
        source_id="official.marketplace",
        source_kind=source_kind,
    )
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.search.web",
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            publisher=publisher,
            version_label="1.0.0",
        ),
        display_label="Web Search",
    )


def test_publisher_metadata_contract_is_frozen_and_extra_forbid() -> None:
    publisher = MarketplacePublisherMetadata(
        publisher_id="acme",
        display_name="Acme Corp",
        website_reference="https://acme.example",
    )
    with pytest.raises(ValidationError):
        publisher.display_name = "Other"


def test_commercial_metadata_free_rejects_price_fields() -> None:
    with pytest.raises(ValidationError, match="FREE commercial_model"):
        MarketplaceCommercialMetadata(
            commercial_model=CommercialModel.FREE,
            minor_units=0,
            currency_code="USD",
        )


def test_commercial_metadata_paid_requires_currency_with_minor_units() -> None:
    with pytest.raises(ValidationError, match="currency_code is required"):
        MarketplaceCommercialMetadata(
            commercial_model=CommercialModel.PAID,
            minor_units=999,
        )


def test_commercial_metadata_paid_accepts_minor_units_and_currency() -> None:
    commercial = MarketplaceCommercialMetadata(
        commercial_model=CommercialModel.PAID,
        minor_units=999,
        currency_code="USD",
        pricing_reference="https://acme.example/pricing",
    )
    assert commercial.minor_units == 999
    assert commercial.currency_code == "USD"


def test_listing_rejects_publisher_mismatch() -> None:
    entry = _entry(publisher="acme")
    with pytest.raises(ValidationError, match="publisher_metadata.publisher_id"):
        MarketplaceCapabilityListing(
            capability=entry,
            publisher_metadata=MarketplacePublisherMetadata(
                publisher_id="other",
                display_name="Other",
            ),
        )


def test_listing_accepts_matching_publisher() -> None:
    entry = _entry(publisher="acme")
    listing = MarketplaceCapabilityListing(
        capability=entry,
        publisher_metadata=MarketplacePublisherMetadata(
            publisher_id="acme",
            display_name="Acme Corp",
        ),
        commercial_metadata=MarketplaceCommercialMetadata(
            commercial_model=CommercialModel.PAID,
            minor_units=500,
            currency_code="EUR",
        ),
    )
    payload = listing.model_dump(mode="json")
    restored = MarketplaceCapabilityListing.model_validate(payload)
    assert restored == listing


def test_listing_serializes_with_schema_version() -> None:
    listing = MarketplaceCapabilityListing(
        listing_id="listing-001",
        capability=_entry(),
    )
    payload = listing.model_dump(mode="json")
    assert payload["schema_version"] == "marketplace_capability_listing.v1"
