# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace federation and air-gapped proof tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.capability_catalog import FederatedCapabilityCatalog, discover_capability_candidates
from intergrax.capability_catalog.adapters.tool import ToolBundleCatalogSource
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import CommercialModel, MarketplaceCommercialMetadata
from intergrax.marketplace import MarketplaceCapabilityCatalogSource, MarketplaceListingRecord, snapshot_without_marketplace
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


def _official_marketplace_tool() -> MarketplaceCapabilityCatalogSource:
    return MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="official.intergrax.marketplace",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tools.marketplace.search",
                display_label="Marketplace Search Tool",
                commercial_metadata=MarketplaceCommercialMetadata(
                    commercial_model=CommercialModel.PAID,
                    minor_units=1200,
                    currency_code="USD",
                ),
            ),
        ),
    )


def _enterprise_marketplace_skill() -> MarketplaceCapabilityCatalogSource:
    return MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="enterprise.acme.marketplace",
            source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.SKILL,
                logical_id="skills.enterprise.analysis",
                publisher="acme",
            ),
        ),
    )


def _register_builtin_tool() -> None:
    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del ctx
        handler = type("H", (), {"execute": lambda self, req: _Out(y=req.input.x)})()
        registry.register(
            ToolContract(
                tool_id="tools.builtin.alpha",
                name="tools.builtin.alpha",
                description="builtin alpha",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            handler,
        )

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="stage11.builtin",
            tool_ids=("tools.builtin.alpha",),
            register=_register,
            description="Stage 11 builtin tools",
        ),
    )


def test_federation_builtin_official_private_marketplace_sources() -> None:
    _register_builtin_tool()
    federated = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            _official_marketplace_tool(),
            _enterprise_marketplace_skill(),
        ),
    )
    snapshot = federated.snapshot()
    kinds = {entry.identity.kind for entry in snapshot.entries}
    source_kinds = {entry.identity.source.source_kind for entry in snapshot.entries}
    assert CapabilityKind.TOOL in kinds
    assert CapabilityKind.SKILL in kinds
    assert CapabilitySourceKind.BUILTIN in source_kinds
    assert CapabilitySourceKind.OFFICIAL in source_kinds
    assert CapabilitySourceKind.ENTERPRISE_PRIVATE in source_kinds
    assert len(snapshot.entries) == 3


def test_same_logical_id_across_sources_preserved() -> None:
    shared_logical_id = "tools.shared.search"
    official = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="official.market",
            source_kind=CapabilitySourceKind.OFFICIAL,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id=shared_logical_id,
            ),
        ),
    )
    private = MarketplaceCapabilityCatalogSource(
        source=CapabilitySourceIdentity(
            source_id="enterprise.market",
            source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
        ),
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id=shared_logical_id,
            ),
        ),
    )
    snapshot = FederatedCapabilityCatalog((official, private)).snapshot()
    matching = [
        entry for entry in snapshot.entries if entry.identity.logical.logical_id == shared_logical_id
    ]
    assert len(matching) == 2
    assert {entry.identity.source.source_kind for entry in matching} == {
        CapabilitySourceKind.OFFICIAL,
        CapabilitySourceKind.ENTERPRISE_PRIVATE,
    }


def test_air_gapped_federation_without_marketplace_sources() -> None:
    _register_builtin_tool()
    federated = FederatedCapabilityCatalog((ToolBundleCatalogSource(),))
    snapshot = snapshot_without_marketplace(federated)
    assert snapshot.entries
    assert all(
        entry.identity.source.source_kind is not CapabilitySourceKind.OFFICIAL
        or entry.identity.source.source_id != "official.intergrax.marketplace"
        for entry in snapshot.entries
    )


def test_marketplace_does_not_mutate_tool_registry() -> None:
    _register_builtin_tool()
    registry = ToolRegistry()
    before = tuple(registry.tool_ids())
    federated = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            _official_marketplace_tool(),
        ),
    )
    discover_capability_candidates(federated.snapshot(), _discovery_query())
    after = tuple(registry.tool_ids())
    assert before == after


class _StaticSource:
    def __init__(self, source_id: str, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._source_id = source_id
        self._entries = entries

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return self._entries
