# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace catalog service tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.capability_catalog import FederatedCapabilityCatalog, StableIdentityRanker, rank_capability_candidates
from intergrax.capability_catalog.adapters.tool import ToolBundleCatalogSource
from intergrax.capability_catalog.candidate import CapabilityDiscoveryCandidate
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.evidence import CapabilityDiscoveryAvailabilityEvidence
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.ranking import CapabilityRankingContext
from intergrax.contracts.marketplace import CommercialModel, MarketplaceCommercialMetadata, MarketplacePublisherMetadata
from intergrax.marketplace import MarketplaceCapabilityCatalogSource, MarketplaceCatalogService, MarketplaceListingRecord
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


@pytest.fixture(autouse=True)
def _isolated_tool_catalog() -> None:
    clear_tool_catalog()
    yield
    clear_tool_catalog()


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _official_tool_source() -> MarketplaceCapabilityCatalogSource:
    return MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="official.intergrax.marketplace",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.marketplace.search",
                display_label="Marketplace Search",
                publisher="intergrax",
                publisher_metadata=MarketplacePublisherMetadata(
                    publisher_id="intergrax",
                    display_name="Intergrax",
                ),
                commercial_metadata=MarketplaceCommercialMetadata(
                    commercial_model=CommercialModel.PAID,
                    minor_units=500,
                    currency_code="USD",
                ),
            ),
        ),
    )


def _enterprise_skill_source() -> MarketplaceCapabilityCatalogSource:
    return MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="enterprise.acme.marketplace",
            source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
        ),
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


def _register_builtin_tool() -> None:
    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del ctx
        handler = type("H", (), {"execute": lambda self, req: _Out(y=req.input.x)})()
        registry.register(
            ToolContract(
                tool_id="tools.builtin.echo",
                name="tools.builtin.echo",
                description="builtin echo",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            handler,
        )

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="stage11.service",
            tool_ids=("tools.builtin.echo",),
            register=_register,
            description="Stage 11 service builtin tools",
        ),
    )


def _service() -> MarketplaceCatalogService:
    _register_builtin_tool()
    official = _official_tool_source()
    enterprise = _enterprise_skill_source()
    catalog = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            official,
            enterprise,
        ),
    )
    return MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(official, enterprise),
    )


def test_product_query_kind_tool_search_returns_marketplace_listing() -> None:
    service = _service()
    views = service.list_listings(
        _discovery_query(kinds=(CapabilityKind.TOOL,)),
        query_text="search",
    )
    assert len(views) == 1
    view = views[0]
    assert view.listing.capability.identity.logical.logical_id == "tools.marketplace.search"
    assert view.listing.publisher_metadata is not None
    assert view.listing.publisher_metadata.publisher_id == "intergrax"
    assert view.listing.commercial_metadata is not None
    assert view.listing.commercial_metadata.commercial_model is CommercialModel.PAID


def test_private_marketplace_listed_with_metadata_preserved() -> None:
    service = _service()
    views = service.list_listings(
        _discovery_query(
            kinds=(CapabilityKind.SKILL,),
            source=None,
        ),
    )
    private_views = [
        view
        for view in views
        if view.listing.capability.identity.source.source_kind
        is CapabilitySourceKind.ENTERPRISE_PRIVATE
    ]
    assert len(private_views) == 1
    assert private_views[0].listing.commercial_metadata is not None
    assert private_views[0].listing.commercial_metadata.commercial_model is CommercialModel.INTERNAL


def test_official_marketplace_listed_with_source_preserved() -> None:
    service = _service()
    views = service.list_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)), query_text="search")
    assert views[0].listing.capability.identity.source.source_kind is CapabilitySourceKind.OFFICIAL


def test_catalog_available_projection_is_not_host_available() -> None:
    service = _service()
    views = service.list_listings(
        _discovery_query(kinds=(CapabilityKind.TOOL,)),
        availability_evidence=CapabilityDiscoveryAvailabilityEvidence(),
        query_text="search",
    )
    assert views[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert views[0].availability is not AvailabilityDisposition.HOST_AVAILABLE


def test_commercial_metadata_does_not_change_ranking_outcome() -> None:
    source = CapabilitySourceIdentity(
        source_id="official.rank.test",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )

    def _candidate(logical_id: str) -> CapabilityDiscoveryCandidate:
        entry = CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=CapabilityKind.TOOL,
                source=source,
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.TOOL,
                    logical_id=logical_id,
                ),
            ),
            provenance=CapabilityProvenance(source=source),
        )
        return CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )

    free = _candidate("tools.rank.alpha")
    paid = _candidate("tools.rank.beta")
    ranker = StableIdentityRanker()
    context = CapabilityRankingContext()
    ranked_free_first = rank_capability_candidates((free, paid), ranker, context=context)
    ranked_paid_first = rank_capability_candidates((paid, free), ranker, context=context)
    assert [item.candidate.identity.logical.logical_id for item in ranked_free_first] == [
        "tools.rank.alpha",
        "tools.rank.beta",
    ]
    assert [item.candidate.identity.logical.logical_id for item in ranked_paid_first] == [
        "tools.rank.alpha",
        "tools.rank.beta",
    ]


def test_list_search_does_not_mutate_registries() -> None:
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    tool_before = tuple(tool_registry.tool_ids())
    skill_before = tuple(skill_registry.skill_ids())
    service = _service()
    service.list_listings(_discovery_query(), query_text="search")
    assert tuple(tool_registry.tool_ids()) == tool_before
    assert tuple(skill_registry.skill_ids()) == skill_before


def test_get_listing_by_identity_key() -> None:
    service = _service()
    key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.intergrax.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.marketplace.search",
    )
    listing = service.get_listing(key)
    assert listing is not None
    assert listing.capability.identity.logical.logical_id == "tools.marketplace.search"
