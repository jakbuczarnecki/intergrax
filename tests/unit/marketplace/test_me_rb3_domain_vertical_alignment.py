# © Artur Czarnecki. All rights reserved.

"""ME-RB3 — Agent / Tool / Skill vertical alignment over common marketplace engine."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.capability_catalog import FederatedCapabilityCatalog
from intergrax.capability_catalog.adapters.agent import AgentCatalogCapabilitySource
from intergrax.capability_catalog.adapters.skill import SkillBundleCatalogSource
from intergrax.capability_catalog.adapters.tool import ToolBundleCatalogSource
from intergrax.contracts.capability_catalog import (
    CapabilityCatalogEntry,
    CapabilityCatalogSource,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace import MarketplaceCapabilityListing, MarketplaceMetadataSource
from intergrax.marketplace import MarketplaceCapabilityCatalogSource, MarketplaceCatalogService
from intergrax.skills.registry.catalog import SkillBundleEntry, clear_skill_catalog, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit

_AGENT_SOURCE = CapabilitySourceIdentity(
    source_id="official-agents",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


class _AgentProvider:
    @property
    def catalog_source_id(self) -> str:
        return _AGENT_SOURCE.source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return [
            AgentCatalogEntry(
                catalog_entry_id="agent.rb3.analyst",
                catalog_source=CatalogSourceIdentity(
                    catalog_source_id=_AGENT_SOURCE.source_id,
                    provider_kind=CatalogProviderKind.OFFICIAL_CATALOG,
                ),
                display_name="RB3 Analyst",
                publisher="intergrax",
                package_id_line="pkg.rb3.analyst",
                version_channel_refs=(),
            ),
        ]


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


class _CatalogSourceWithProductMetadata:
    """Wire domain ``CapabilityCatalogSource`` + ``MarketplaceMetadataSource`` on one source_id."""

    def __init__(
        self,
        inner: CapabilityCatalogSource,
        *,
        declared_source: CapabilitySourceIdentity,
    ) -> None:
        self._inner = inner
        self._declared_source = declared_source

    @property
    def source_id(self) -> str:
        return self._inner.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        return self._declared_source

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return self._inner.read_entries()

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        listings: list[MarketplaceCapabilityListing] = []
        for entry in self.read_entries():
            listings.append(
                MarketplaceCapabilityListing(
                    listing_id=f"listing-{entry.identity.logical.logical_id}",
                    capability=entry,
                    publisher_metadata=None,
                    commercial_metadata=None,
                ),
            )
        return tuple(listings)


class _CustomToolVerticalSource:
    """External tool catalog provider — no subclass of default tool adapter."""

    @property
    def source_id(self) -> str:
        return "partner.tools.catalog"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        from intergrax.contracts.capability_catalog import (
            CapabilityDiscoveryIdentity,
            CapabilityLogicalIdentity,
            CapabilityProvenance,
        )

        partner = CapabilitySourceIdentity(
            source_id=self.source_id,
            source_kind=CapabilitySourceKind.THIRD_PARTY,
        )
        return (
            CapabilityCatalogEntry(
                identity=CapabilityDiscoveryIdentity(
                    kind=CapabilityKind.TOOL,
                    source=partner,
                    logical=CapabilityLogicalIdentity(
                        kind=CapabilityKind.TOOL,
                        logical_id="tools.partner.custom",
                    ),
                ),
                provenance=CapabilityProvenance(
                    source=partner,
                    version_label="1.0.0",
                    publisher="partner",
                ),
                display_label="Partner Custom Tool",
            ),
        )


class _CustomToolMarketplaceMetadataSource:
    """External marketplace metadata for partner tool catalog — no default subclass."""

    def __init__(self, catalog: CapabilityCatalogSource) -> None:
        self._catalog = catalog

    @property
    def source_id(self) -> str:
        return self._catalog.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        entries = self._catalog.read_entries()
        if not entries:
            raise AssertionError("partner catalog must expose at least one entry")
        return entries[0].identity.source

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        return tuple(
            MarketplaceCapabilityListing(
                listing_id=f"partner-listing-{entry.identity.logical.logical_id}",
                capability=entry,
                publisher_metadata=None,
                commercial_metadata=None,
            )
            for entry in self._catalog.read_entries()
        )


@pytest.fixture(autouse=True)
def _isolated_domain_catalogs() -> None:
    clear_tool_catalog()
    clear_skill_catalog()
    yield
    clear_tool_catalog()
    clear_skill_catalog()


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _register_rb3_tool_and_skill() -> None:
    def _register_tool(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del ctx
        handler = type("H", (), {"execute": lambda self, req: _Out(y=req.input.x)})()
        registry.register(
            ToolContract(
                tool_id="tools.rb3.echo",
                name="tools.rb3.echo",
                description="rb3 echo",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            handler,
        )

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="rb3.tools",
            tool_ids=("tools.rb3.echo",),
            register=_register_tool,
            description="RB3 tool bundle",
        ),
    )

    def _register_skill(registry: SkillRegistry) -> None:
        del registry

    register_skill_bundle(
        SkillBundleEntry(
            bundle_id="rb3.skills",
            skill_ids=("skills.rb3.summarize",),
            register=_register_skill,
            description="RB3 skill bundle",
        ),
    )


def _vertical_sources() -> tuple[_CatalogSourceWithProductMetadata, ...]:
    _register_rb3_tool_and_skill()
    agent = _CatalogSourceWithProductMetadata(
        AgentCatalogCapabilitySource(_AgentProvider()),
        declared_source=_AGENT_SOURCE,
    )
    from intergrax.capability_catalog.adapters.tool import TOOL_BUILTIN_CATALOG_SOURCE_ID

    tool_inner = ToolBundleCatalogSource()
    tool = _CatalogSourceWithProductMetadata(
        tool_inner,
        declared_source=CapabilitySourceIdentity(
            source_id=TOOL_BUILTIN_CATALOG_SOURCE_ID,
            source_kind=CapabilitySourceKind.BUILTIN,
        ),
    )
    from intergrax.capability_catalog.adapters.skill import SKILL_BUILTIN_CATALOG_SOURCE_ID

    skill_inner = SkillBundleCatalogSource()
    skill = _CatalogSourceWithProductMetadata(
        skill_inner,
        declared_source=CapabilitySourceIdentity(
            source_id=SKILL_BUILTIN_CATALOG_SOURCE_ID,
            source_kind=CapabilitySourceKind.BUILTIN,
        ),
    )
    return (agent, tool, skill)


def test_common_marketplace_supports_agent_tool_skill_verticals() -> None:
    agent, tool, skill = _vertical_sources()
    federated = FederatedCapabilityCatalog((agent, tool, skill))
    snapshot = federated.snapshot()
    kinds = {entry.identity.kind for entry in snapshot.entries}
    assert kinds == {CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL}

    identity_keys = [entry.identity.sort_key for entry in snapshot.entries]
    assert len(identity_keys) == len(set(identity_keys))
    assert list(identity_keys) == sorted(identity_keys)

    service = MarketplaceCatalogService(
        catalog=federated,
        marketplace_sources=(agent, tool, skill),
    )
    for kind in (CapabilityKind.AGENT, CapabilityKind.TOOL, CapabilityKind.SKILL):
        views = service.list_listings(_discovery_query(kinds=(kind,)))
        assert len(views) == 1
        assert views[0].listing.capability.identity.kind is kind
        assert views[0].listing.listing_id is not None


def test_federated_agent_tool_skill_with_custom_tool_provider() -> None:
    agent, tool, skill = _vertical_sources()
    custom: CapabilityCatalogSource = _CustomToolVerticalSource()
    federated = FederatedCapabilityCatalog((agent, tool, skill, custom))
    snapshot = federated.snapshot()
    logical_ids = {entry.identity.logical.logical_id for entry in snapshot.entries}
    assert "tools.partner.custom" in logical_ids
    assert len({entry.identity.kind for entry in snapshot.entries}) == 3


def test_custom_vertical_provider_appears_in_common_marketplace() -> None:
    agent, tool, skill = _vertical_sources()
    custom_catalog: CapabilityCatalogSource = _CustomToolVerticalSource()
    custom_metadata: MarketplaceMetadataSource = _CustomToolMarketplaceMetadataSource(
        custom_catalog,
    )

    assert not isinstance(custom_catalog, ToolBundleCatalogSource)
    assert not isinstance(custom_metadata, MarketplaceCapabilityCatalogSource)

    federated = FederatedCapabilityCatalog((agent, tool, skill, custom_catalog))
    snapshot = federated.snapshot()
    partner_entry = next(
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id == "tools.partner.custom"
    )
    assert partner_entry.identity.kind is CapabilityKind.TOOL
    assert partner_entry.identity.source.source_id == "partner.tools.catalog"
    assert partner_entry.provenance.publisher == "partner"
    assert partner_entry.provenance.version_label == "1.0.0"

    service = MarketplaceCatalogService(
        catalog=federated,
        marketplace_sources=(agent, tool, skill, custom_metadata),
    )

    identity_key = CapabilityIdentityKey.from_discovery_identity(partner_entry.identity)
    listing = service.get_listing(identity_key)
    assert listing is not None
    assert listing.listing_id == "partner-listing-tools.partner.custom"
    assert listing.capability == partner_entry
    assert listing.capability.identity.kind is CapabilityKind.TOOL
    assert listing.capability.identity.source.source_id == custom_catalog.source_id
    assert listing.capability.identity.logical.logical_id == "tools.partner.custom"

    tool_views = service.list_listings(_discovery_query(kinds=(CapabilityKind.TOOL,)))
    partner_views = [
        view
        for view in tool_views
        if view.listing.capability.identity.logical.logical_id == "tools.partner.custom"
    ]
    assert len(partner_views) == 1
    assert partner_views[0].listing.capability == partner_entry


def test_vertical_adapters_do_not_import_marketplace_or_nexus() -> None:
    adapters_root = _package_root("intergrax.capability_catalog.adapters")
    vertical_modules = ("agent.py", "tool.py", "skill.py")
    forbidden_prefixes = (
        "intergrax.marketplace",
        "intergrax.nexus",
        "intergrax.runtime.nexus",
        "intergrax.tools.registry.runtime",
        "intergrax.skills.registry.runtime",
    )
    for module_name in vertical_modules:
        path = adapters_root / module_name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in forbidden_prefixes:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{module_name} imports forbidden dependency: {imported}",
                    )


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_agent_tool_skill_sources_expose_capability_catalog_source_shape() -> None:
    agent, tool, skill = _vertical_sources()
    for source in (agent, tool, skill):
        assert isinstance(source.source_id, str)
        entries = source.read_entries()
        assert isinstance(entries, tuple)
        assert all(isinstance(entry, CapabilityCatalogEntry) for entry in entries)


def test_marketplace_listing_read_does_not_mutate_tool_or_skill_catalogs() -> None:
    agent, tool, skill = _vertical_sources()
    from intergrax.tools.registry.catalog import iter_bundles
    from intergrax.skills.registry.catalog import iter_bundles as iter_skill_bundles

    before_tools = tuple(iter_bundles())
    before_skills = tuple(iter_skill_bundles())
    service = MarketplaceCatalogService(
        catalog=FederatedCapabilityCatalog((agent, tool, skill)),
        marketplace_sources=(agent, tool, skill),
    )
    _ = service.list_listings(_discovery_query())
    assert tuple(iter_bundles()) == before_tools
    assert tuple(iter_skill_bundles()) == before_skills


def test_get_listing_by_identity_uses_common_service_for_each_vertical() -> None:
    agent, tool, skill = _vertical_sources()
    federated = FederatedCapabilityCatalog((agent, tool, skill))
    service = MarketplaceCatalogService(
        catalog=federated,
        marketplace_sources=(agent, tool, skill),
    )
    for kind, logical_id in (
        (CapabilityKind.AGENT, "agent.rb3.analyst"),
        (CapabilityKind.TOOL, "tools.rb3.echo"),
        (CapabilityKind.SKILL, "skills.rb3.summarize"),
    ):
        entry = next(
            e
            for e in federated.snapshot().entries
            if e.identity.kind is kind and e.identity.logical.logical_id == logical_id
        )
        listing = service.get_listing(
            CapabilityIdentityKey.from_discovery_identity(entry.identity),
        )
        assert listing is not None
        assert listing.capability.identity.kind is kind
