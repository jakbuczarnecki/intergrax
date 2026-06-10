# © Artur Czarnecki. All rights reserved.

"""Integration marketplace catalog with trust scoring (AUDIT-IDEAL-13.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.integrations.registry.catalog import catalog_snapshot, iter_entries


class IntegrationMarketplaceEntry(BaseModel):
    slug: str
    status: IntegrationStatus
    trust_score: float = Field(ge=0.0, le=1.0)
    categories: list[str] = Field(default_factory=list)


class IntegrationMarketplaceCatalog(BaseModel):
    schema_version: str = "1.0.0"
    entries: list[IntegrationMarketplaceEntry] = Field(default_factory=list)


def trust_score_for_status(status: IntegrationStatus) -> float:
    if status is IntegrationStatus.STABLE:
        return 1.0
    if status is IntegrationStatus.BETA:
        return 0.75
    return 0.35


def build_integration_marketplace_catalog() -> IntegrationMarketplaceCatalog:
    """Build trust-scored marketplace view from the live integration catalog."""
    entries = [
        IntegrationMarketplaceEntry(
            slug=entry.slug,
            status=entry.status,
            trust_score=trust_score_for_status(entry.status),
            categories=[category.value for category in entry.categories],
        )
        for entry in sorted(iter_entries(), key=lambda item: item.slug)
    ]
    if not entries and catalog_snapshot():
        entries = [
            IntegrationMarketplaceEntry(
                slug=slug,
                status=entry.status,
                trust_score=trust_score_for_status(entry.status),
                categories=[category.value for category in entry.categories],
            )
            for slug, entry in sorted(catalog_snapshot().items())
        ]
    return IntegrationMarketplaceCatalog(entries=entries)
