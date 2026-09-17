# © Artur Czarnecki. All rights reserved.

"""ME-17 — Marketplace production qualification (failure, concurrency, isolation, boundaries)."""

from __future__ import annotations

import ast
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    CapabilityDiscoveryCandidate,
    DefaultCatalogEntryTextSearchStrategy,
    FederatedCapabilityCatalog,
    RankedCapabilityCandidate,
    SnapshotCachingCapabilityCatalog,
    StableIdentityRanker,
    govern_capability_candidates,
    rank_capability_candidates,
)
from intergrax.capability_catalog.errors import CapabilityGovernanceEvaluatorUnavailableError
from intergrax.capability_catalog.federation import CapabilityCatalogFederationPolicy
from intergrax.capability_catalog.governance import CapabilityGovernanceDecision
from intergrax.capability_catalog.snapshot import CapabilityCatalogFederationCompleteness
from intergrax.capability_catalog.snapshot_cache import BoundedInMemoryCapabilityCatalogSnapshotCache
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityCatalogEntry as ContractCatalogEntry,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityNeed,
    CapabilityProvenance,
    CapabilityRankingEvidence,
    CapabilityRankingSignal,
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
    CapabilityReleaseIdentity,
    CapabilitySearchEvidence,
    CapabilitySearchSignal,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_metering import CapabilityUsageEvent, CapabilityUsageKind
from intergrax.contracts.marketplace import (
    CapabilityDiscoveryTraceFacts,
    CapabilityHandoffConsumerError,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffEnvelope,
    CapabilityHandoffIdentityConflictError,
    CapabilityMarketplaceExplicitSelection,
    MarketplaceCapabilityListing,
    MarketplaceListingProjection,
    MarketplaceListingRecord,
    MarketplaceMetadataSource,
    MarketplaceQueryContext,
    MarketplaceVisibility,
)
from intergrax.contracts.marketplace.acquisition import (
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
    SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
    MachineCapabilityAcquisitionRequest,
)
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
from intergrax.contracts.marketplace.visibility import MarketplaceVisibilityScope
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.diagnostics import (
    InMemoryMarketplaceDiagnosticObserver,
    MarketplacePipelineObservationSession,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
)
from intergrax.marketplace.observed_pipeline import run_marketplace_intelligence_pipeline
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest, DynamicToolAcquisitionService
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider, _CustomMe14ToolCatalogProvider
from testing_support.me15_skill_catalog_provider import _CustomMe15SkillCatalogProvider

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.me17",
    source_kind=CapabilitySourceKind.OFFICIAL,
)

_ME17_PERF: dict[str, dict[str, float]] = {}


def _entry(source_id: str, logical_id: str) -> CapabilityCatalogEntry:
    source = CapabilitySourceIdentity(
        source_id=source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    kind = CapabilityKind.TOOL
    return CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=kind,
            source=source,
            logical=CapabilityLogicalIdentity(kind=kind, logical_id=logical_id),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
        display_label=logical_id,
    )


def _global_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _isolation_service() -> MarketplaceCatalogService:
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="public-tool",
            display_label="public",
            publisher="intergrax",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tenant-a-private",
            display_label="private-a",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
                tenant_id="tenant-a",
            ),
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.SKILL,
            logical_id="org-a-private",
            display_label="org-a",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
                organization_id="org-a",
            ),
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _StaticSource:
    def __init__(self, source_id: str, entries: tuple[CapabilityCatalogEntry, ...]) -> None:
        self._source_id = source_id
        self._entries = entries

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        return self._entries


class _FailingSource:
    @property
    def source_id(self) -> str:
        return "zzz.failing"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        raise CapabilityCatalogSourceFailure("expected outage")


class _BuggySource:
    @property
    def source_id(self) -> str:
        return "buggy"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        raise RuntimeError("defect")


class _BrokenRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.me17"

    def rank(self, candidates, context):
        del candidates, context
        raise RuntimeError("ranker programming defect")


class _UnavailableGovernance:
    @property
    def evaluator_id(self) -> str:
        return "gov.unavailable.me17"

    def evaluate(self, candidate: RankedCapabilityCandidate, context: CapabilityGovernanceContext):
        del candidate, context
        raise CapabilityGovernanceEvaluatorUnavailableError("evaluator down")


class _CustomCatalogSource:
    @property
    def source_id(self) -> str:
        return _OFFICIAL.source_id

    def read_entries(self) -> tuple[ContractCatalogEntry, ...]:
        return (
            ContractCatalogEntry(
                identity=CapabilityDiscoveryIdentity(
                    kind=CapabilityKind.TOOL,
                    source=_OFFICIAL,
                    logical=CapabilityLogicalIdentity(
                        kind=CapabilityKind.TOOL,
                        logical_id="tools.me17.custom",
                    ),
                ),
                provenance=CapabilityProvenance(source=_OFFICIAL, version_label="1.0.0"),
                display_label="Custom",
            ),
        )


class _CustomListingProjection:
    @property
    def projection_id(self) -> str:
        return "custom.me17.projection"

    def project_catalog_entry(self, source, record):
        logical_id = record.logical_id.strip()
        return CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=record.kind,
                source=source,
                logical=CapabilityLogicalIdentity(kind=record.kind, logical_id=logical_id),
            ),
            provenance=CapabilityProvenance(
                source=source,
                version_label=record.version_label,
                package_reference=record.package_reference,
                content_digest=record.content_digest,
                publisher=record.publisher,
            ),
            display_label=record.display_label or logical_id,
        )

    def build_listing(self, source, record) -> MarketplaceCapabilityListing:
        return MarketplaceCapabilityListing(
            listing_id=record.listing_id or "listing-me17",
            capability=self.project_catalog_entry(source, record),
        )


class _CustomMetadataSource:
    def __init__(self) -> None:
        self._projection = _CustomListingProjection()
        self._record = MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tools.me17.custom",
            version_label="1.0.0",
            publisher="intergrax",
            listing_id="listing-me17-custom",
        )

    @property
    def source_id(self) -> str:
        return _OFFICIAL.source_id

    @property
    def source(self) -> CapabilitySourceIdentity:
        return _OFFICIAL

    def read_listings(self) -> tuple[MarketplaceCapabilityListing, ...]:
        return (self._projection.build_listing(self.source, self._record),)


class _ReverseRanker:
    @property
    def ranker_id(self) -> str:
        return "custom.reverse.me17"

    def rank(self, candidates, context):
        del context
        ordered = tuple(reversed(candidates))
        return tuple(
            RankedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilityRankingEvidence(
                    ranker_id=self.ranker_id,
                    rank_position=index,
                    signal=CapabilityRankingSignal.STABLE_IDENTITY_ORDER,
                ),
            )
            for index, candidate in enumerate(ordered, start=1)
        )


class _SingleRecommendation:
    @property
    def recommendation_strategy_id(self) -> str:
        return "custom.single.me17"

    def recommend(self, governed, context):
        del context
        from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation

        pick = governed[0]
        return (
            CapabilityRecommendation(
                governed=pick,
                evidence=CapabilityRecommendationEvidence(
                    recommendation_strategy_id=self.recommendation_strategy_id,
                    reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
                    reason_text="me17 pick",
                    rank_position=pick.ranking_evidence.rank_position,
                ),
            ),
        )


class _AgentOnlySearch:
    @property
    def search_strategy_id(self) -> str:
        return "custom.agent_only.me17"

    def search(self, candidates, query, context):
        del query, context
        from intergrax.capability_catalog import SearchedCapabilityCandidate

        return tuple(
            SearchedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilitySearchEvidence(
                    search_strategy_id=self.search_strategy_id,
                    signal=CapabilitySearchSignal.PASS_THROUGH,
                ),
            )
            for candidate in candidates
            if candidate.identity.kind is CapabilityKind.AGENT
        )


# --- Required failure tests ---


def test_me17_source_failure_respects_snapshot_policy() -> None:
    federated = FederatedCapabilityCatalog(
        (_StaticSource("aaa.ok", (_entry("aaa.ok", "ok"),)), _FailingSource()),
    )
    with pytest.raises(CapabilityCatalogSourceFailure):
        federated.snapshot(federation_policy=CapabilityCatalogFederationPolicy.STRICT_COMPLETE)
    partial = federated.snapshot(federation_policy=CapabilityCatalogFederationPolicy.ALLOW_PARTIAL)
    assert partial.federation_completeness == CapabilityCatalogFederationCompleteness.PARTIAL


def test_me17_programming_source_failure_propagates() -> None:
    with pytest.raises(RuntimeError, match="defect"):
        FederatedCapabilityCatalog((_BuggySource(),)).snapshot()


def test_me17_broken_ranker_programming_failure_propagates() -> None:
    service = _isolation_service()
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries
    )
    discovery = MarketplaceDiscoveryService(
        search_strategy=DefaultCatalogEntryTextSearchStrategy(),
        ranker=_BrokenRanker(),
    )
    with pytest.raises(RuntimeError, match="ranker programming defect"):
        discovery.search_and_rank(candidates)


def test_me17_governance_evaluator_failure_is_fail_closed() -> None:
    service = _isolation_service()
    snapshot = service._catalog.snapshot()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in snapshot.entries[:1]
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    result = govern_capability_candidates(
        ranked,
        evaluators=(_UnavailableGovernance(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    assert not result.allowed
    assert any(
        item.reason_code is CapabilityGovernanceReasonCode.EVALUATOR_FAILURE
        for item in result.blocked[0].evidence
    )


def test_me17_lifecycle_consumer_failure_is_not_silent() -> None:
    class _FailingConsumer:
        @property
        def consumer_id(self) -> str:
            return "consumer-me17"

        def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
            raise CapabilityHandoffConsumerError("downstream failed")

    release = CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="tools.x"),
        ),
        publisher="pub",
        version_label="1.0.0",
        content_digest="sha256:x",
    )
    envelope = CapabilityHandoffEnvelope(
        handoff_id="handoff-me17-fail",
        tenant_id="tenant-a",
        selected_release=release,
        discovery_correlation_id="disc-me17",
        selection_id="sel-me17",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id="consumer-me17",
        discovery_trace=CapabilityDiscoveryTraceFacts(
            discovery_correlation_id="disc-me17",
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-a"),
            visible_candidate_count=1,
            governed_admissible_count=1,
        ),
        explicit_selection=CapabilityMarketplaceExplicitSelection(
            selection_id="sel-me17",
            discovery_correlation_id="disc-me17",
            selected_release=release,
            selector_id="selector-me17",
        ),
        recorded_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
    )
    delivery = CapabilityHandoffDeliveryService(
        consumer=_FailingConsumer(),
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
    )
    with pytest.raises(CapabilityHandoffConsumerError, match="downstream failed"):
        delivery.deliver(envelope)


# --- Required isolation tests ---


def test_me17_tenant_private_catalog_never_leaks_cross_tenant() -> None:
    service = _isolation_service()
    tenant_a = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-a"),
    )
    tenant_b = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-b"),
    )
    a_ids = {v.listing.capability.identity.logical.logical_id for v in tenant_a}
    b_ids = {v.listing.capability.identity.logical.logical_id for v in tenant_b}
    assert "tenant-a-private" in a_ids
    assert "tenant-a-private" not in b_ids


def test_me17_organization_private_catalog_never_leaks_cross_org() -> None:
    service = _isolation_service()
    org_a = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(organization_id="org-a"),
    )
    org_b = service.list_listings(
        _global_query(),
        marketplace_query_context=MarketplaceQueryContext(organization_id="org-b"),
    )
    assert any(v.listing.capability.identity.logical.logical_id == "org-a-private" for v in org_a)
    assert not any(v.listing.capability.identity.logical.logical_id == "org-a-private" for v in org_b)


def test_me17_cache_never_crosses_visibility_scope() -> None:
    cache = BoundedInMemoryCapabilityCatalogSnapshotCache(max_entries=8)
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="public-tool",
            publisher="intergrax",
        ),
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tenant-a-private",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
                tenant_id="tenant-a",
            ),
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    inner = FederatedCapabilityCatalog((source,))
    catalog = SnapshotCachingCapabilityCatalog(inner, cache=cache)
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    tenant_a = MarketplaceQueryContext(tenant_id="tenant-a")
    tenant_b = MarketplaceQueryContext(tenant_id="tenant-b")
    a_ids = {
        v.listing.capability.identity.logical.logical_id
        for v in service.list_listings(_global_query(), marketplace_query_context=tenant_a)
    }
    b_ids = {
        v.listing.capability.identity.logical.logical_id
        for v in service.list_listings(_global_query(), marketplace_query_context=tenant_b)
    }
    assert "tenant-a-private" in a_ids
    assert "tenant-a-private" not in b_ids


def test_me17_diagnostics_never_cross_visibility_scope() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    records = (
        MarketplaceListingRecord(
            kind=CapabilityKind.AGENT,
            logical_id="agents.secret.foreign",
            publisher="intergrax",
            visibility=MarketplaceVisibility(
                scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
                tenant_id="tenant-b",
            ),
        ),
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    session = MarketplacePipelineObservationSession.for_discovery(
        "me17-diag-iso",
        observer=observer,
    )
    run_marketplace_intelligence_pipeline(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
        discovery_query=_global_query(),
        marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-a"),
        observation=session,
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
    )
    serialized = " ".join(str(event.model_dump()) for event in observer.events)
    assert "agents.secret.foreign" not in serialized
    assert "tenant-b" not in serialized


# --- Required concurrency tests ---


def test_me17_concurrent_catalog_queries_are_isolated() -> None:
    records = tuple(
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id=f"tool.{index}",
            publisher="intergrax",
        )
        for index in range(200)
    )
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
    results: list[tuple[str, ...]] = []
    errors: list[BaseException] = []

    def _run(tenant: str) -> None:
        try:
            views = service.list_listings(
                _global_query(),
                marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant),
            )
            results.append(
                tuple(v.listing.capability.identity.logical.logical_id for v in views),
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(f"tenant-{index}",)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert not errors
    assert len(results) == 16
    assert all(result == results[0] for result in results)


def test_me17_concurrent_private_scope_queries_do_not_leak() -> None:
    service = _isolation_service()
    leaks: list[str] = []
    lock = threading.Lock()

    def _run(tenant: str) -> None:
        views = service.list_listings(
            _global_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant),
        )
        ids = {v.listing.capability.identity.logical.logical_id for v in views}
        if tenant != "tenant-a" and "tenant-a-private" in ids:
            with lock:
                leaks.append(tenant)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_run, f"tenant-{index}") for index in range(12)]
        for future in as_completed(futures):
            future.result()
    assert leaks == []


def test_me17_concurrent_duplicate_handoffs_are_idempotent() -> None:
    class _RecordingConsumer:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def consumer_id(self) -> str:
            return "consumer-me17-dedupe"

        def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
            self.calls += 1

    consumer = _RecordingConsumer()
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
    )
    release = CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="tools.dedupe"),
        ),
        publisher="pub",
        version_label="1.0.0",
        content_digest="sha256:dedupe",
    )
    envelope = CapabilityHandoffEnvelope(
        handoff_id="handoff-me17-concurrent",
        tenant_id="tenant-a",
        selected_release=release,
        discovery_correlation_id="disc-dedupe",
        selection_id="sel-dedupe",
        consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        downstream_consumer_id=consumer.consumer_id,
        discovery_trace=CapabilityDiscoveryTraceFacts(
            discovery_correlation_id="disc-dedupe",
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-a"),
            visible_candidate_count=1,
            governed_admissible_count=1,
        ),
        explicit_selection=CapabilityMarketplaceExplicitSelection(
            selection_id="sel-dedupe",
            discovery_correlation_id="disc-dedupe",
            selected_release=release,
            selector_id="selector-dedupe",
        ),
        recorded_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
    )
    barrier = threading.Barrier(2)
    dispositions: list[CapabilityHandoffDeliveryDisposition] = []

    def _deliver() -> None:
        barrier.wait()
        result = delivery.deliver(envelope)
        dispositions.append(result.disposition)

    threads = [threading.Thread(target=_deliver) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert consumer.calls == 1
    assert CapabilityHandoffDeliveryDisposition.DELIVERED in dispositions


def _me17_tool_acquisition_stack() -> tuple[DynamicToolAcquisitionService, ToolHostLifecycleService]:
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-me17")
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry({provider.catalog_source_id: provider}),
        activation=lifecycle,
        materializer=Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    return service, lifecycle


def test_me17_concurrent_tool_acquisition_is_safe() -> None:
    from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate

    service, lifecycle = _me17_tool_acquisition_stack()
    provider = Me14ToolCatalogProvider()
    identity = ToolDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=provider.list_entries()[0].package_reference,
            package_version=ME14_VERSION_V1,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    identity_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id=provider.catalog_source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=ME14_TOOL_LOGICAL_ID,
    )
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _acquire() -> None:
        try:
            service.acquire(
                DynamicToolAcquisitionRequest(
                    operation_id="op-me17-concurrent",
                    host_profile_id="host-me17",
                    capability_identity_key=identity_key,
                    selected_identity=identity,
                ),
            )
        except BaseException as exc:
            with lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_acquire) for _ in range(8)]
        for future in as_completed(futures):
            future.result()
    assert not errors
    assert lifecycle.is_active(ME14_TOOL_LOGICAL_ID)


def test_me17_concurrent_skill_binding_is_safe() -> None:
    from intergrax.skills.catalog import SkillCatalogProviderRegistry
    from intergrax.skills.dynamic_acquisition import (
        DynamicSkillAcquisitionRequest,
        DynamicSkillAcquisitionService,
    )
    from intergrax.skills.host_lifecycle import SkillHostLifecycleService
    from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
    from testing_support.canonical_me15_reference_skill import ME15_DIGEST_V1, ME15_SKILL_LOGICAL_ID, ME15_VERSION_V1
    from testing_support.me15_skill_catalog_provider import Me15SkillCatalogProvider
    from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer

    provider = Me15SkillCatalogProvider()
    lifecycle = SkillHostLifecycleService(host_profile_id="host-me17-skill")
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=lifecycle,
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    entry = provider.list_entries()[0]
    identity = SkillDiscoveryCandidateIdentity(
        catalog_source_id=provider.catalog_source_id,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference=entry.package_reference,
            package_version=ME15_VERSION_V1,
            package_digest=ME15_DIGEST_V1,
        ),
    )
    identity_key = CapabilityIdentityKey(
        kind=CapabilityKind.SKILL,
        source_id=provider.catalog_source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=ME15_SKILL_LOGICAL_ID,
    )
    errors: list[BaseException] = []

    def _bind() -> None:
        try:
            service.acquire(
                DynamicSkillAcquisitionRequest(
                    operation_id="skill-op-me17-concurrent",
                    host_profile_id="host-me17-skill",
                    capability_identity_key=identity_key,
                    selected_identity=identity,
                ),
            )
        except BaseException as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_bind) for _ in range(6)]
        for future in as_completed(futures):
            future.result()
    assert not errors
    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_me17_cross_vertical_concurrent_lifecycle_is_safe(tmp_path: Path) -> None:
    from testing_support.marketplace_mixed_capability_execution_composition import (
        MarketplaceMixedCapabilityProofStack,
    )

    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _agent() -> None:
        try:
            stack.handoff_agent(
                discovery_correlation_id="cv-agent",
                selection_id="sel-a",
                handoff_id="h-a-cv",
            )
        except BaseException as exc:
            with lock:
                errors.append(exc)

    def _tool() -> None:
        try:
            stack.handoff_tool(
                discovery_correlation_id="cv-tool",
                selection_id="sel-t",
                handoff_id="h-t-cv",
            )
        except BaseException as exc:
            with lock:
                errors.append(exc)

    def _skill() -> None:
        try:
            stack.handoff_skill(
                discovery_correlation_id="cv-skill",
                selection_id="sel-s",
                handoff_id="h-s-cv",
            )
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=_agent),
        threading.Thread(target=_tool),
        threading.Thread(target=_skill),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert not errors


# --- Pluginability ---


def test_me17_custom_catalog_source() -> None:
    catalog = FederatedCapabilityCatalog((_CustomCatalogSource(),))
    assert len(catalog.snapshot().entries) == 1


def test_me17_custom_metadata_source() -> None:
    metadata: MarketplaceMetadataSource = _CustomMetadataSource()
    catalog_source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(metadata._record,),
        listing_projection=_CustomListingProjection(),
    )
    catalog = FederatedCapabilityCatalog((catalog_source,))
    service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(metadata,),
    )
    key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id=_OFFICIAL.source_id,
        source_kind=_OFFICIAL.source_kind,
        logical_id="tools.me17.custom",
    )
    listing = service.get_listing(key)
    assert listing is not None
    assert listing.listing_id == "listing-me17-custom"


def test_me17_custom_listing_projection() -> None:
    projection: MarketplaceListingProjection = _CustomListingProjection()
    record = MarketplaceListingRecord(kind=CapabilityKind.TOOL, logical_id="tools.proj")
    listing = projection.build_listing(_OFFICIAL, record)
    assert listing.capability.identity.logical.logical_id == "tools.proj"


def test_me17_custom_ranker() -> None:
    service = _isolation_service()
    candidates = tuple(
        CapabilityDiscoveryCandidate(
            catalog_entry=entry,
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        )
        for entry in service._catalog.snapshot().entries
    )
    ranked = rank_capability_candidates(candidates, _ReverseRanker())
    assert ranked[0].evidence.ranker_id == "custom.reverse.me17"


def test_me17_custom_governance_evaluator() -> None:
    class _DenyAll:
        @property
        def evaluator_id(self) -> str:
            return "deny.all.me17"

        def evaluate(self, candidate, context):
            del context
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
                ),
            )

    ranked = rank_capability_candidates(
        (
            CapabilityDiscoveryCandidate(
                catalog_entry=_entry("x", "tool.one"),
                availability=AvailabilityDisposition.CATALOG_AVAILABLE,
            ),
        ),
        StableIdentityRanker(),
    )
    result = govern_capability_candidates(
        ranked,
        evaluators=(_DenyAll(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    )
    assert not result.allowed


def test_me17_custom_recommendation_strategy() -> None:
    candidates = (
        CapabilityDiscoveryCandidate(
            catalog_entry=_entry("x", "tool.one"),
            availability=AvailabilityDisposition.CATALOG_AVAILABLE,
        ),
    )
    ranked = rank_capability_candidates(candidates, StableIdentityRanker())
    governed = govern_capability_candidates(
        ranked,
        evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
    ).allowed
    recs = MarketplaceRecommendationService(recommendation_strategy=_SingleRecommendation()).recommend(
        governed,
        recommendation_context=CapabilityRecommendationContext(top_n=3),
    )
    assert recs[0].evidence.recommendation_strategy_id == "custom.single.me17"


def test_me17_custom_tool_provider() -> None:
    custom = _CustomMe14ToolCatalogProvider()
    assert custom.catalog_source_id == "custom.me14.provider"
    assert custom.list_entries()


def test_me17_custom_skill_provider() -> None:
    custom = _CustomMe15SkillCatalogProvider()
    assert custom.catalog_source_id == "custom.me15.provider"
    assert custom.list_entries()


# --- Architecture boundary gates ---


def _marketplace_py_files() -> list[Path]:
    package = importlib.import_module("intergrax.marketplace")
    root = Path(package.__path__[0])
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def test_me17_marketplace_core_has_no_nexus_dependency() -> None:
    forbidden = ("intergrax.nexus", "intergrax.runtime.nexus")
    for path in _marketplace_py_files():
        if "handoff" in path.parts and "adapters" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                            raise AssertionError(f"{path.name} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        raise AssertionError(f"{path.name} imports {node.module}")


def test_me17_marketplace_core_has_no_execution_dependency() -> None:
    forbidden = ("intergrax.runtime.execution", "intergrax.runtime.nexus")
    for path in _marketplace_py_files():
        if "handoff" in path.parts and "adapters" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            if mod and any(mod == p or mod.startswith(f"{p}.") for p in forbidden):
                raise AssertionError(f"{path.name} imports {mod}")


def test_me17_marketplace_core_has_no_domain_registry_implementation_dependency() -> None:
    forbidden = (
        "intergrax.tools.registry.runtime",
        "intergrax.skills.registry.runtime",
        "intergrax.agent_distribution",
    )
    for path in _marketplace_py_files():
        if "handoff" in path.parts and "adapters" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    if node.module == prefix or node.module.startswith(f"{prefix}."):
                        raise AssertionError(f"{path.name} imports {node.module}")


def test_me17_production_has_no_testing_support_imports() -> None:
    roots = (
        Path(importlib.import_module("intergrax.marketplace").__path__[0]).parent,
    )
    violations: list[str] = []
    for root_name in ("marketplace", "capability_catalog"):
        root = roots[0] / root_name
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "testing_support" or node.module.startswith("testing_support."):
                        violations.append(path.as_posix())
    assert violations == []


# --- Contract tests ---


def test_me17_public_contracts_reject_invalid_identity() -> None:
    with pytest.raises(Exception):
        MachineCapabilityAcquisitionRequest(
            request_id="   ",
            need=CapabilityNeed(kinds=(CapabilityKind.TOOL,)),
            discovery_query=_global_query(),
        )


def test_me17_public_contract_schema_ids_are_unique() -> None:
    ids = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
        SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    )
    assert len(ids) == len(set(ids))


def test_me17_usage_event_contains_no_pricing_or_settlement() -> None:
    payload = CapabilityUsageEvent.model_json_schema()
    props = payload.get("properties", {})
    forbidden = ("price", "pricing", "settlement", "invoice", "billing")
    for name in props:
        lowered = name.lower()
        assert not any(token in lowered for token in forbidden)


def test_me17_skill_binding_is_not_usage() -> None:
    kinds = {item.value for item in CapabilityUsageKind}
    assert "binding" not in kinds
    assert "acquisition" not in kinds


# --- Mixed regression ---


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter
    from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

    install_diagnostic_cursor_secret(monkeypatch)
    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override if agent_override is not None else adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_me17_agent_tool_skill_mixed_flow_still_executes(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    import asyncio

    from testing_support.marketplace_mixed_capability_execution_composition import (
        MarketplaceMixedCapabilityProofStack,
    )
    from testing_support.me16_mixed_harness_execution import (
        execute_me16_mixed_via_host_execution_engine,
    )

    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    stack.run_all_handoffs(
        discovery_correlation_id="me17-mixed",
        agent_handoff_id="h-ma",
        tool_handoff_id="h-mt",
        skill_handoff_id="h-ms",
    )
    assert stack.assert_execution_readiness().execution_allowed
    asyncio.run(
        execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "exec",
        ),
    )


# --- Performance baseline (report-only, no SLA) ---


def _synthetic_records(count: int) -> tuple[MarketplaceListingRecord, ...]:
    return tuple(
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id=f"tools.perf.{index:05d}",
            publisher="intergrax",
        )
        for index in range(count)
    )


def test_me17_performance_baseline_snapshot_and_discovery() -> None:
    for size in (100, 1000):
        records = _synthetic_records(size)
        source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
        catalog = FederatedCapabilityCatalog((source,))
        service = MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))
        discovery = MarketplaceDiscoveryService.with_defaults()
        t0 = time.perf_counter()
        snapshot = catalog.snapshot()
        snap_s = time.perf_counter() - t0
        candidates = tuple(
            CapabilityDiscoveryCandidate(
                catalog_entry=entry,
                availability=AvailabilityDisposition.CATALOG_AVAILABLE,
            )
            for entry in snapshot.entries
        )
        t1 = time.perf_counter()
        ranked = discovery.search_and_rank(candidates, search_query=None)
        disc_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        govern_capability_candidates(
            ranked,
            evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            context=CapabilityGovernanceContext(posture=CapabilityGovernancePosture.STRICT),
        )
        gov_s = time.perf_counter() - t2
        _ME17_PERF[str(size)] = {
            "snapshot_s": snap_s,
            "discovery_s": disc_s,
            "rank_governance_s": gov_s,
        }
        assert len(snapshot.entries) == size
        assert len(service.list_listings(_global_query())) == size


def test_me17_handoff_identity_conflict_fail_closed() -> None:
    admission = InMemoryCapabilityHandoffDeliveryAdmission()
    release_a = CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_OFFICIAL,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="tools.a"),
        ),
        publisher="pub",
        version_label="1.0.0",
        content_digest="sha256:a",
    )
    release_b = release_a.model_copy(update={"content_digest": "sha256:b"})
    base_kwargs = {
        "handoff_id": "handoff-conflict",
        "tenant_id": "tenant-a",
        "discovery_correlation_id": "disc-c",
        "selection_id": "sel-c",
        "consumer_target": CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
        "downstream_consumer_id": "consumer-c",
        "discovery_trace": CapabilityDiscoveryTraceFacts(
            discovery_correlation_id="disc-c",
            marketplace_query_context=MarketplaceQueryContext(tenant_id="tenant-a"),
            visible_candidate_count=1,
            governed_admissible_count=1,
        ),
        "recorded_at": datetime(2026, 3, 17, tzinfo=timezone.utc),
    }

    class _NoopConsumer:
        @property
        def consumer_id(self) -> str:
            return "consumer-c"

        def consume(self, envelope: CapabilityHandoffEnvelope) -> None:
            del envelope

    delivery = CapabilityHandoffDeliveryService(
        consumer=_NoopConsumer(),
        delivery_admission=admission,
    )
    env_a = CapabilityHandoffEnvelope(
        **base_kwargs,
        selected_release=release_a,
        explicit_selection=CapabilityMarketplaceExplicitSelection(
            selection_id="sel-c",
            discovery_correlation_id="disc-c",
            selected_release=release_a,
            selector_id="selector-c",
        ),
    )
    delivery.deliver(env_a)
    env_b = CapabilityHandoffEnvelope(
        **base_kwargs,
        selected_release=release_b,
        explicit_selection=CapabilityMarketplaceExplicitSelection(
            selection_id="sel-c",
            discovery_correlation_id="disc-c",
            selected_release=release_b,
            selector_id="selector-c",
        ),
    )
    with pytest.raises(CapabilityHandoffIdentityConflictError):
        delivery.deliver(env_b)


def test_me17_catalog_identity_conflict_within_source_fail_closed() -> None:
    base = _entry("official.me17", "tools.conflict")
    conflicting = base.model_copy(update={"display_label": "Different"})
    catalog = FederatedCapabilityCatalog(
        (_StaticSource("official.me17", (base, conflicting)),),
    )
    with pytest.raises(CapabilityCatalogIdentityConflict):
        catalog.snapshot()
