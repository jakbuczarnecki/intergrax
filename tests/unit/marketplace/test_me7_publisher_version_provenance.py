# © Artur Czarnecki. All rights reserved.

"""ME-7 publisher, version, and provenance integrity proofs."""

from __future__ import annotations

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityDiscoveryCandidate,
    FederatedCapabilityCatalog,
    govern_capability_candidates,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.errors import CapabilityCatalogIdentityConflict
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityCatalogEntry,
    CapabilityCatalogSource,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilityRecommendationContext,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    MarketplaceListingRecord,
    marketplace_capability_selection,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)

pytestmark = pytest.mark.unit

_PUBLISHER_A = "publisher-a"
_PUBLISHER_B = "publisher-b"
_LOGICAL_SEARCH = "tools.search"
_DIGEST_A = "sha256:aaa"
_DIGEST_B = "sha256:bbb"


def _source(source_id: str) -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _entry(
    *,
    source: CapabilitySourceIdentity,
    logical_id: str = _LOGICAL_SEARCH,
    kind: CapabilityKind = CapabilityKind.TOOL,
    publisher: str | None = None,
    version_label: str | None = None,
    content_digest: str | None = None,
    package_reference: str | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(
            source=source,
            publisher=publisher,
            version_label=version_label,
            content_digest=content_digest,
            package_reference=package_reference,
        ),
        display_label=logical_id,
    )


def _release(entry: CapabilityCatalogEntry) -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity.from_catalog_entry(entry)


def _pipeline_to_recommendation(
    catalog: FederatedCapabilityCatalog,
    marketplace_source: MarketplaceCapabilityCatalogSource,
    *,
    logical_id: str,
) -> CapabilityCatalogEntry:
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    views = service.list_listings(_discovery_query())
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=view.listing.capability,
            availability=view.availability,
        )
        for view in views
    )
    discovery = MarketplaceDiscoveryService.with_defaults()
    recommendation = MarketplaceRecommendationService.with_defaults()
    ranked = discovery.search_and_rank(candidates)
    governed = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    ).allowed
    picks = recommendation.recommend(
        governed,
        recommendation_context=CapabilityRecommendationContext(top_n=10),
    )
    for pick in picks:
        entry = pick.governed.ranked.candidate.catalog_entry
        if entry.identity.logical.logical_id == logical_id:
            return entry
    raise AssertionError(f"no recommendation for logical_id {logical_id!r}")


def _mixed_vertical_bundle() -> tuple[FederatedCapabilityCatalog, MarketplaceCapabilityCatalogSource]:
    official = _source("official.me7.marketplace")
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.AGENT,
            logical_id="agents.me7.alpha",
            publisher="agent-publisher",
            version_label="1.0.0",
            content_digest="sha256:agent",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tools.me7.beta",
            publisher="tool-publisher",
            version_label="2.0.0",
            content_digest="sha256:tool",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.SKILL,
            logical_id="skills.me7.gamma",
            publisher="skill-publisher",
            version_label="3.0.0",
            content_digest="sha256:skill",
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=official, records=records)
    return FederatedCapabilityCatalog((source,)), source


def _entry_by_logical_id(
    catalog: FederatedCapabilityCatalog,
    logical_id: str,
) -> CapabilityCatalogEntry:
    for entry in catalog.snapshot().entries:
        if entry.identity.logical.logical_id == logical_id:
            return entry
    raise AssertionError(f"missing catalog entry for {logical_id!r}")


def test_publisher_identity_is_preserved_end_to_end() -> None:
    catalog, marketplace_source = _mixed_vertical_bundle()
    expected = _entry_by_logical_id(catalog, "tools.me7.beta")
    final = _pipeline_to_recommendation(
        catalog,
        marketplace_source,
        logical_id="tools.me7.beta",
    )
    assert _release(final).publisher == _release(expected).publisher == "tool-publisher"


def test_version_identity_is_preserved_end_to_end() -> None:
    catalog, marketplace_source = _mixed_vertical_bundle()
    expected = _entry_by_logical_id(catalog, "tools.me7.beta")
    final = _pipeline_to_recommendation(
        catalog,
        marketplace_source,
        logical_id="tools.me7.beta",
    )
    assert _release(final).version_label == _release(expected).version_label == "2.0.0"


def test_provenance_is_preserved_end_to_end() -> None:
    catalog, marketplace_source = _mixed_vertical_bundle()
    expected = _entry_by_logical_id(catalog, "tools.me7.beta")
    final = _pipeline_to_recommendation(
        catalog,
        marketplace_source,
        logical_id="tools.me7.beta",
    )
    assert final.provenance == expected.provenance
    assert final.identity.source == expected.identity.source


def test_different_sources_same_logical_remain_distinct() -> None:
    source_a = _source("official.publisher-a")
    source_b = _source("official.publisher-b")
    entry_a = _entry(
        source=source_a,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    entry_b = _entry(
        source=source_b,
        publisher=_PUBLISHER_B,
        version_label="1.0.0",
        content_digest=_DIGEST_B,
    )
    snapshot = FederatedCapabilityCatalog(
        (
            _StaticSource(source_a.source_id, (entry_a,)),
            _StaticSource(source_b.source_id, (entry_b,)),
        ),
    ).snapshot()
    releases = {_release(entry).release_sort_key for entry in snapshot.entries}
    assert len(releases) == 2


def test_distinct_logical_rows_expose_distinct_discovery_entries() -> None:
    """Distinct logical_id values are distinct discovery identities (not multi-version registry)."""
    source = _source("official.distinct-logical-rows")
    v1 = _entry(
        source=source,
        logical_id="tools.search.v1",
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    v2 = _entry(
        source=source,
        logical_id="tools.search.v2",
        publisher=_PUBLISHER_A,
        version_label="2.0.0",
        content_digest=_DIGEST_B,
    )
    snapshot = FederatedCapabilityCatalog(
        (_StaticSource(source.source_id, (v1, v2)),),
    ).snapshot()
    assert len(snapshot.entries) == 2


def test_same_source_same_logical_different_version_conflicts() -> None:
    source = _source("official.one-release-per-discovery")
    v1 = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    v2 = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="2.0.0",
        content_digest=_DIGEST_B,
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        FederatedCapabilityCatalog((_StaticSource(source.source_id, (v1, v2)),)).snapshot()


def test_same_source_same_logical_different_publisher_conflicts() -> None:
    source = _source("official.publisher-collision")
    entry_a = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    entry_b = _entry(
        source=source,
        publisher=_PUBLISHER_B,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        merge_capability_catalog_entries(
            (
                (source.source_id, entry_a),
                (source.source_id, entry_b),
            ),
        )


def test_same_source_same_logical_same_release_dedupes() -> None:
    source = _source("official.dedupe")
    entry = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    merged = merge_capability_catalog_entries(
        (
            (source.source_id, entry),
            (source.source_id, entry),
        ),
    )
    assert len(merged) == 1
    assert _release(merged[0]) == _release(entry)


def test_same_source_same_logical_different_package_reference_conflicts() -> None:
    source = _source("official.package-ref-collision")
    first = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
        package_reference="pkg:search@1.0.0",
    )
    second = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
        package_reference="pkg:search@1.0.0+build2",
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        merge_capability_catalog_entries(
            (
                (source.source_id, first),
                (source.source_id, second),
            ),
        )


def test_release_identity_is_parallel_to_discovery_identity() -> None:
    source = _source("official.release-vs-discovery")
    entry = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="3.0.0",
        content_digest=_DIGEST_A,
    )
    release = _release(entry)
    assert release.discovery.sort_key == entry.identity.sort_key
    assert release.release_sort_key != entry.identity.sort_key
    assert release.release_sort_key[:4] == entry.identity.sort_key


def test_current_release_provenance_preserved_end_to_end() -> None:
    test_provenance_is_preserved_end_to_end()


def test_conflicting_same_release_integrity_fails_closed() -> None:
    source = _source("official.conflict")
    base = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_A,
    )
    conflicting = _entry(
        source=source,
        publisher=_PUBLISHER_A,
        version_label="1.0.0",
        content_digest=_DIGEST_B,
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        merge_capability_catalog_entries(
            (
                (source.source_id, base),
                (source.source_id, conflicting),
            ),
        )


class _CustomCatalogSource:
    @property
    def source_id(self) -> str:
        return "custom.me7.source"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        source = _source(self.source_id)
        return (
            _entry(
                source=source,
                logical_id="tools.custom.release",
                publisher="external-publisher",
                version_label="7.7.7",
                content_digest="sha256:custom",
            ),
        )


def test_custom_catalog_source_preserves_release_provenance_without_core_changes() -> None:
    source: CapabilityCatalogSource = _CustomCatalogSource()
    entry = FederatedCapabilityCatalog((source,)).snapshot().entries[0]
    release = _release(entry)
    assert release.publisher == "external-publisher"
    assert release.version_label == "7.7.7"
    assert release.content_digest == "sha256:custom"
    assert entry.identity.source.source_id == "custom.me7.source"


def test_mixed_agent_tool_skill_release_provenance_pipeline() -> None:
    catalog, marketplace_source = _mixed_vertical_bundle()
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    )
    discovery = MarketplaceDiscoveryService.with_defaults()
    recommendation = MarketplaceRecommendationService.with_defaults()
    for view in service.list_listings(_discovery_query()):
        entry = view.listing.capability
        candidates = (
            CapabilityDiscoveryCandidate(
                catalog_entry=entry,
                availability=AvailabilityDisposition.CATALOG_AVAILABLE,
            ),
        )
        ranked = discovery.search_and_rank(candidates)
        governed = govern_capability_candidates(
            ranked,
            evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
        ).allowed
        picks = recommendation.recommend(governed)
        assert picks
        final = picks[0].governed.ranked.candidate.catalog_entry
        assert _release(final) == _release(entry)


def test_lifecycle_selection_carries_canonical_provenance() -> None:
    catalog, marketplace_source = _mixed_vertical_bundle()
    entry = _entry_by_logical_id(catalog, "agents.me7.alpha")
    listing = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(marketplace_source,),
    ).list_listings(_discovery_query())[0].listing
    selection = marketplace_capability_selection(
        listing_id=listing.listing_id or "listing-1",
        capability=listing.capability,
    )
    assert selection.capability.provenance == entry.provenance


class _StaticSource:
    def __init__(self, source_id: str, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._source_id = source_id
        self._entries = entries

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return self._entries
