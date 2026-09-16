# © Artur Czarnecki. All rights reserved.

"""ME-12 machine capability acquisition API qualification."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityCatalogEntry,
    CapabilityCatalogSnapshot,
    CapabilityCatalogFederationCompleteness,
    CapabilityGovernanceDecision,
    FederatedCapabilityCatalog,
    RankedCapabilityCandidate,
    StableIdentityRanker,
)
from intergrax.capability_catalog.errors import CapabilityCatalogSourceFailure
from intergrax.capability_catalog.federation import CapabilityCatalogFederationPolicy
from intergrax.capability_catalog.governance import CapabilityGovernanceEvaluator
from intergrax.contracts.capability_catalog import (
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
    CapabilityRecommendationContext,
    CapabilityRecommendationEvidence,
    CapabilityRecommendationReasonCode,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.marketplace import (
    CapabilityHandoffConsumerTarget,
    CapabilityHandoffDeliveryDisposition,
    CapabilityHandoffDeliveryResult,
    MarketplaceListingRecord,
    MarketplaceObservationContext,
    MarketplaceQueryContext,
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.contracts.marketplace.acquisition import (
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
    SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
    MachineCapabilityAcquisitionHandoffRequest,
    MachineCapabilityAcquisitionHandoffResponse,
    MachineCapabilityAcquisitionOutcome,
    MachineCapabilityAcquisitionPolicy,
    MachineCapabilityAcquisitionRequest,
    MachineCapabilityRecommendation,
    MachineCatalogFederationCompleteness,
)
from intergrax.contracts.marketplace.acquisition import (
    MachineCapabilityAcquisitionSelection,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import (
    MachineCapabilityAcquisitionPolicyError,
    MachineCapabilityAcquisitionSelectionError,
    MachineCapabilityAcquisitionService,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.me12",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
        **kwargs,
    )


def _need(**kwargs: object) -> CapabilityNeed:
    return CapabilityNeed(**kwargs)


def _acquire_request(
    *,
    request_id: str = "req-1",
    context: MarketplaceQueryContext | None = None,
    correlation_id: str = "discovery-corr-me12",
    kinds: tuple[CapabilityKind, ...] = (),
) -> MachineCapabilityAcquisitionRequest:
    return MachineCapabilityAcquisitionRequest(
        request_id=request_id,
        need=_need(kinds=kinds),
        discovery_query=_discovery_query(),
        marketplace_query_context=context or MarketplaceQueryContext(),
        observation=MarketplaceObservationContext(
            discovery_correlation_id=correlation_id,
        ),
        recommendation_context=CapabilityRecommendationContext(top_n=10),
    )


def _public_record(kind: CapabilityKind, logical_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
    )


def _tenant_private(kind: CapabilityKind, logical_id: str, tenant_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        ),
    )


def _org_private(kind: CapabilityKind, logical_id: str, organization_id: str) -> MarketplaceListingRecord:
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
            organization_id=organization_id,
        ),
    )


def _catalog_service(*records: MarketplaceListingRecord) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _DenyLogicalGovernance:
    @property
    def evaluator_id(self) -> str:
        return "test.deny.logical"

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        del context
        if candidate.identity.logical.logical_id == "blocked-agent":
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
                ),
            )
        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )


class _RecordingHandoffConsumer:
    def __init__(self) -> None:
        self.envelopes: list = []

    @property
    def consumer_id(self) -> str:
        return "machine.consumer.me12"

    def consume(self, envelope) -> None:
        self.envelopes.append(envelope)


def _machine_service(
    catalog: MarketplaceCatalogService,
    consumer: _RecordingHandoffConsumer,
    *,
    discovery_service: MarketplaceDiscoveryService | None = None,
    recommendation_service: MarketplaceRecommendationService | None = None,
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...] | None = None,
    acquisition_policy: MachineCapabilityAcquisitionPolicy | None = None,
) -> MachineCapabilityAcquisitionService:
    trace = InMemoryCapabilityHandoffTraceEvidenceConsumer()
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=trace,
    )
    orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=catalog,
        discovery_service=discovery_service or MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=governance_evaluators
        or (AvailabilityPreservingGovernanceEvaluator(), _DenyLogicalGovernance()),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        delivery_service=delivery,
    )
    return MachineCapabilityAcquisitionService(
        catalog_service=catalog,
        discovery_service=discovery_service or MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=governance_evaluators
        or (AvailabilityPreservingGovernanceEvaluator(), _DenyLogicalGovernance()),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        recommendation_service=recommendation_service
        or MarketplaceRecommendationService.with_defaults(),
        handoff_orchestrator=orchestrator,
        acquisition_policy=acquisition_policy,
    )


def _mixed_fixture_service() -> tuple[
    MarketplaceCatalogService,
    MachineCapabilityAcquisitionService,
    _RecordingHandoffConsumer,
]:
    consumer = _RecordingHandoffConsumer()
    catalog = _catalog_service(
        _public_record(CapabilityKind.AGENT, "public-agent"),
        _tenant_private(CapabilityKind.TOOL, "tenant-a-tool", "tenant-a"),
        _org_private(CapabilityKind.SKILL, "org-a-skill", "org-a"),
        _public_record(CapabilityKind.AGENT, "blocked-agent"),
    )
    service = _machine_service(catalog, consumer)
    return catalog, service, consumer


def _recommended_logical_ids(response) -> set[str]:
    return {item.release.discovery.logical.logical_id for item in response.recommendations}


def test_machine_acquisition_api_is_contract_driven_and_pluginable() -> None:
    module = importlib.import_module("intergrax.contracts.marketplace.acquisition")
    assert hasattr(module, "MachineCapabilityAcquisitionPolicy")
    service_module = importlib.import_module("intergrax.marketplace.acquisition.service")
    tree = ast.parse(Path(service_module.__file__).read_text(encoding="utf-8"))
    assert "MachineCapabilityAcquisitionService" in {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }


def test_machine_request_without_scope_returns_only_public_governed_recommendations() -> None:
    _, service, _consumer = _mixed_fixture_service()
    response = service.acquire(_acquire_request())
    assert response.outcome is MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE
    ids = _recommended_logical_ids(response)
    assert "public-agent" in ids
    assert "tenant-a-tool" not in ids
    assert "org-a-skill" not in ids
    assert "blocked-agent" not in ids


def test_machine_tenant_scope_never_leaks_foreign_private_capability() -> None:
    _, service, _consumer = _mixed_fixture_service()
    response = service.acquire(
        _acquire_request(context=MarketplaceQueryContext(tenant_id="tenant-a")),
    )
    ids = _recommended_logical_ids(response)
    assert "tenant-a-tool" in ids
    assert "public-agent" in ids
    assert not any("tenant-b" in item for item in ids)


def test_machine_org_scope_never_leaks_foreign_private_capability() -> None:
    _, service, _consumer = _mixed_fixture_service()
    response = service.acquire(
        _acquire_request(context=MarketplaceQueryContext(organization_id="org-a")),
    )
    ids = _recommended_logical_ids(response)
    assert "org-a-skill" in ids
    assert "public-agent" in ids


def test_machine_response_contains_only_governed_candidates() -> None:
    _, service, _consumer = _mixed_fixture_service()
    response = service.acquire(_acquire_request())
    for item in response.recommendations:
        assert isinstance(item, MachineCapabilityRecommendation)
        assert item.governance_evidence_refs
        assert item.release.discovery.logical.logical_id
        assert item.ranking_strategy_id


class _SpySearch:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def search_strategy_id(self) -> str:
        return "spy.search.me12"

    def search(self, candidates, query, context):
        del query, context
        self.calls += 1
        from intergrax.capability_catalog.search import SearchedCapabilityCandidate
        from intergrax.contracts.capability_catalog.search import CapabilitySearchEvidence, CapabilitySearchSignal

        return tuple(
            SearchedCapabilityCandidate(
                candidate=candidate,
                evidence=CapabilitySearchEvidence(
                    search_strategy_id=self.search_strategy_id,
                    signal=CapabilitySearchSignal.PASS_THROUGH,
                ),
            )
            for candidate in candidates
        )


class _SpyRanker:
    def __init__(self) -> None:
        self.calls = 0
        self._inner = StableIdentityRanker()

    @property
    def ranker_id(self) -> str:
        return self._inner.ranker_id

    def rank(self, searched, context):
        self.calls += 1
        return self._inner.rank(searched, context)


class _SpyRecommender:
    def __init__(self) -> None:
        self.calls = 0
        from intergrax.capability_catalog.recommendation import (
            DefaultTopRankedCapabilityRecommendationStrategy,
        )

        self._inner = DefaultTopRankedCapabilityRecommendationStrategy()

    @property
    def recommendation_strategy_id(self) -> str:
        return self._inner.recommendation_strategy_id

    def recommend(self, governed_candidates, context):
        self.calls += 1
        effective = context or CapabilityRecommendationContext(top_n=10)
        return self._inner.recommend(governed_candidates, context=effective)


def test_machine_api_uses_configured_search_rank_and_recommendation_strategies() -> None:
    consumer = _RecordingHandoffConsumer()
    catalog = _catalog_service(_public_record(CapabilityKind.AGENT, "public-agent"))
    search = _SpySearch()
    ranker = _SpyRanker()
    recommender = _SpyRecommender()
    discovery = MarketplaceDiscoveryService(search_strategy=search, ranker=ranker)
    recommendation = MarketplaceRecommendationService(recommendation_strategy=recommender)
    service = _machine_service(
        catalog,
        consumer,
        discovery_service=discovery,
        recommendation_service=recommendation,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    service.acquire(_acquire_request())
    assert search.calls == 1
    assert ranker.calls == 1
    assert recommender.calls == 1


def test_machine_acquisition_is_deterministic_for_same_snapshot_and_context() -> None:
    _, service, _consumer = _mixed_fixture_service()
    request = _acquire_request()
    first = service.acquire(request)
    second = service.acquire(request)
    first_ids = [item.release.discovery.logical.logical_id for item in first.recommendations]
    second_ids = [item.release.discovery.logical.logical_id for item in second.recommendations]
    assert first_ids == second_ids


def test_machine_no_match_has_typed_outcome() -> None:
    consumer = _RecordingHandoffConsumer()
    catalog = _catalog_service()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    response = service.acquire(_acquire_request())
    assert response.outcome is MachineCapabilityAcquisitionOutcome.NO_MATCH
    assert response.recommendations == ()


def test_machine_selection_must_reference_recommended_exact_release() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    acquire = service.acquire(_acquire_request(correlation_id="corr-select-ok"))
    rec = acquire.recommendations[0]
    release = rec.release
    handoff = service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=_acquire_request(correlation_id="corr-select-ok"),
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-1",
                discovery_correlation_id="corr-select-ok",
                selected_release=release,
                selector_id="machine.client",
            ),
            handoff_id="handoff-1",
        ),
    )
    assert handoff.delivery.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert len(consumer.envelopes) == 1


def test_machine_selection_rejects_non_recommended_capability() -> None:
    catalog = _catalog_service(
        _public_record(CapabilityKind.AGENT, "agent-a"),
        _public_record(CapabilityKind.AGENT, "agent-b"),
    )
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    acquire = service.acquire(
        MachineCapabilityAcquisitionRequest(
            request_id="req-reject",
            need=_need(kinds=(CapabilityKind.AGENT,)),
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(),
            observation=MarketplaceObservationContext(
                discovery_correlation_id="corr-reject",
            ),
            recommendation_context=CapabilityRecommendationContext(top_n=1),
        ),
    )
    recommended_id = acquire.recommendations[0].release.discovery.logical.logical_id
    snapshot = catalog._catalog.snapshot()
    other = next(
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id != recommended_id
    )
    release = CapabilityReleaseIdentity.from_catalog_entry(other)
    narrow_request = MachineCapabilityAcquisitionRequest(
        request_id="req-reject",
        need=_need(kinds=(CapabilityKind.AGENT,)),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        observation=MarketplaceObservationContext(
            discovery_correlation_id="corr-reject",
        ),
        recommendation_context=CapabilityRecommendationContext(top_n=1),
    )
    with pytest.raises(MachineCapabilityAcquisitionSelectionError, match="recommendation"):
        service.select_and_handoff(
            MachineCapabilityAcquisitionHandoffRequest(
                acquisition_request=narrow_request,
                selection=MachineCapabilityAcquisitionSelection(
                    selection_id="sel-bad",
                    discovery_correlation_id="corr-reject",
                    selected_release=release,
                    selector_id="machine.client",
                ),
                handoff_id="handoff-bad",
            ),
        )


def test_machine_selection_rejects_foreign_private_capability() -> None:
    catalog = _catalog_service(
        _public_record(CapabilityKind.TOOL, "public-tool"),
        _tenant_private(CapabilityKind.TOOL, "tenant-b-tool", "tenant-b"),
    )
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    snapshot = catalog._catalog.snapshot()
    private_entry = next(
        e for e in snapshot.entries if e.identity.logical.logical_id == "tenant-b-tool"
    )
    release = CapabilityReleaseIdentity.from_catalog_entry(private_entry)
    with pytest.raises(MachineCapabilityAcquisitionSelectionError):
        service.select_and_handoff(
            MachineCapabilityAcquisitionHandoffRequest(
                acquisition_request=_acquire_request(
                    context=MarketplaceQueryContext(tenant_id="tenant-a"),
                    correlation_id="corr-foreign",
                ),
                selection=MachineCapabilityAcquisitionSelection(
                    selection_id="sel-foreign",
                    discovery_correlation_id="corr-foreign",
                    selected_release=release,
                    selector_id="machine.client",
                ),
                handoff_id="handoff-foreign",
            ),
        )


def test_machine_selection_rejects_governance_blocked_capability() -> None:
    catalog, service, _consumer = _mixed_fixture_service()
    snapshot = catalog._catalog.snapshot()
    blocked_entry = next(
        entry for entry in snapshot.entries if entry.identity.logical.logical_id == "blocked-agent"
    )
    release = CapabilityReleaseIdentity.from_catalog_entry(blocked_entry)
    with pytest.raises(MachineCapabilityAcquisitionSelectionError, match="recommendation"):
        service.select_and_handoff(
            MachineCapabilityAcquisitionHandoffRequest(
                acquisition_request=_acquire_request(correlation_id="corr-blocked"),
                selection=MachineCapabilityAcquisitionSelection(
                    selection_id="sel-blocked",
                    discovery_correlation_id="corr-blocked",
                    selected_release=release,
                    selector_id="machine.client",
                ),
                handoff_id="handoff-blocked",
            ),
        )


def test_stale_selection_is_revalidated_against_current_marketplace_state() -> None:
    catalog = _catalog_service(
        _public_record(CapabilityKind.TOOL, "tool-release-v1"),
        _public_record(CapabilityKind.TOOL, "tool-release-v2"),
    )
    service = _machine_service(
        catalog,
        _RecordingHandoffConsumer(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    narrow_request = MachineCapabilityAcquisitionRequest(
        request_id="req-stale",
        need=_need(kinds=(CapabilityKind.TOOL,)),
        discovery_query=_discovery_query(),
        observation=MarketplaceObservationContext(discovery_correlation_id="corr-stale"),
        recommendation_context=CapabilityRecommendationContext(top_n=1),
    )
    acquire = service.acquire(narrow_request)
    selected = acquire.recommendations[0]
    snapshot = catalog._catalog.snapshot()
    other_entry = next(
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id != selected.release.discovery.logical.logical_id
    )
    release_other = CapabilityReleaseIdentity.from_catalog_entry(other_entry)
    with pytest.raises(MachineCapabilityAcquisitionSelectionError, match="recommendation"):
        service.select_and_handoff(
            MachineCapabilityAcquisitionHandoffRequest(
                acquisition_request=narrow_request,
                selection=MachineCapabilityAcquisitionSelection(
                    selection_id="sel-stale",
                    discovery_correlation_id="corr-stale",
                    selected_release=release_other,
                    selector_id="machine.client",
                ),
                handoff_id="handoff-stale",
            ),
        )


def test_machine_selection_preserves_exact_release_identity() -> None:
    catalog = _catalog_service(
        _public_record(CapabilityKind.TOOL, "tool-release-v1"),
        _public_record(CapabilityKind.TOOL, "tool-release-v2"),
    )
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    narrow_request = MachineCapabilityAcquisitionRequest(
        request_id="req-version",
        need=_need(kinds=(CapabilityKind.TOOL,)),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        observation=MarketplaceObservationContext(discovery_correlation_id="corr-version"),
        recommendation_context=CapabilityRecommendationContext(top_n=1),
    )
    acquire = service.acquire(narrow_request)
    selected = acquire.recommendations[0]
    release_selected = selected.release
    service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=narrow_request,
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-v1",
                discovery_correlation_id="corr-version",
                selected_release=release_selected,
                selector_id="machine.client",
            ),
            handoff_id="handoff-v1",
        ),
    )
    envelope = consumer.envelopes[0]
    assert envelope.selected_release == release_selected
    snapshot = catalog._catalog.snapshot()
    other_entry = next(
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id
        != selected.release.discovery.logical.logical_id
    )
    release_other = CapabilityReleaseIdentity.from_catalog_entry(other_entry)
    with pytest.raises(MachineCapabilityAcquisitionSelectionError, match="recommendation"):
        service.select_and_handoff(
            MachineCapabilityAcquisitionHandoffRequest(
                acquisition_request=narrow_request,
                selection=MachineCapabilityAcquisitionSelection(
                    selection_id="sel-v2-stale",
                    discovery_correlation_id="corr-version",
                    selected_release=release_other,
                    selector_id="machine.client",
                ),
                handoff_id="handoff-v2",
            ),
        )


@pytest.mark.parametrize(
    ("logical_id", "kind", "target"),
    (
        ("public-agent", CapabilityKind.AGENT, CapabilityHandoffConsumerTarget.AGENT_DOMAIN),
        ("tenant-a-tool", CapabilityKind.TOOL, CapabilityHandoffConsumerTarget.TOOL_DOMAIN),
        ("org-a-skill", CapabilityKind.SKILL, CapabilityHandoffConsumerTarget.SKILL_DOMAIN),
    ),
)
def test_machine_handoff_uses_existing_domain_lifecycle_contract(
    logical_id: str,
    kind: CapabilityKind,
    target: CapabilityHandoffConsumerTarget,
) -> None:
    consumer = _RecordingHandoffConsumer()
    catalog = _catalog_service(
        _public_record(CapabilityKind.AGENT, "public-agent"),
        _tenant_private(CapabilityKind.TOOL, "tenant-a-tool", "tenant-a"),
        _org_private(CapabilityKind.SKILL, "org-a-skill", "org-a"),
    )
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    context = MarketplaceQueryContext()
    if logical_id == "tenant-a-tool":
        context = MarketplaceQueryContext(tenant_id="tenant-a")
    if logical_id == "org-a-skill":
        context = MarketplaceQueryContext(organization_id="org-a")
    correlation = f"corr-{logical_id}"
    acquire = service.acquire(
        _acquire_request(context=context, correlation_id=correlation),
    )
    rec = next(
        item
        for item in acquire.recommendations
        if item.release.discovery.logical.logical_id == logical_id
    )
    release = rec.release
    service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=_acquire_request(context=context, correlation_id=correlation),
            selection=MachineCapabilityAcquisitionSelection(
                selection_id=f"sel-{logical_id}",
                discovery_correlation_id=correlation,
                selected_release=release,
                selector_id="machine.client",
                consumer_target=target,
            ),
            handoff_id=f"handoff-{logical_id}",
        ),
    )
    envelope = consumer.envelopes[0]
    assert envelope.consumer_target is target
    assert envelope.selected_release.discovery.kind is kind


def test_machine_acquisition_does_not_execute_capability() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    correlation = "corr-no-exec"
    response = service.acquire(_acquire_request(correlation_id=correlation))
    assert "execution" not in response.model_dump_json()
    rec = response.recommendations[0]
    release = rec.release
    handoff = service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=_acquire_request(correlation_id=correlation),
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-no-exec",
                discovery_correlation_id=correlation,
                selected_release=release,
                selector_id="machine.client",
            ),
            handoff_id="handoff-no-exec",
        ),
    )
    assert handoff.delivery.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED
    assert "execute" not in handoff.model_dump_json().lower()


def test_machine_acquisition_correlation_reaches_handoff() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    correlation = "corr-trace-me12"
    acquire = service.acquire(_acquire_request(correlation_id=correlation))
    rec = acquire.recommendations[0]
    release = rec.release
    service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=_acquire_request(correlation_id=correlation),
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-trace",
                discovery_correlation_id=correlation,
                selected_release=release,
                selector_id="machine.client",
            ),
            handoff_id="handoff-trace",
        ),
    )
    assert consumer.envelopes[0].discovery_correlation_id == correlation


class _BrokenRanker:
    @property
    def ranker_id(self) -> str:
        return "broken.ranker"

    def rank(self, searched, context):
        del searched, context
        raise RuntimeError("ranker programming defect")


def test_unexpected_machine_pipeline_programming_error_propagates() -> None:
    consumer = _RecordingHandoffConsumer()
    catalog = _catalog_service(_public_record(CapabilityKind.AGENT, "public-agent"))
    discovery = MarketplaceDiscoveryService(
        search_strategy=_SpySearch(),
        ranker=_BrokenRanker(),
    )
    service = _machine_service(
        catalog,
        consumer,
        discovery_service=discovery,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        service.acquire(_acquire_request())


class _ToolOnlyPolicy:
    def narrow_recommendations(
        self,
        recommendations: tuple[MachineCapabilityRecommendation, ...],
    ) -> tuple[MachineCapabilityRecommendation, ...]:
        return tuple(
            item for item in recommendations if item.release.discovery.kind is CapabilityKind.TOOL
        )


class _EvilWidenPolicy:
    def narrow_recommendations(
        self,
        recommendations: tuple[MachineCapabilityRecommendation, ...],
    ) -> tuple[MachineCapabilityRecommendation, ...]:
        if not recommendations:
            return recommendations
        forged_entry = CapabilityCatalogEntry(
            identity=CapabilityDiscoveryIdentity(
                kind=CapabilityKind.AGENT,
                source=_OFFICIAL,
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.AGENT,
                    logical_id="forged-agent",
                ),
            ),
            provenance=CapabilityProvenance(source=_OFFICIAL, version_label="9.9.9"),
            display_label="forged",
        )
        forged = MachineCapabilityRecommendation(
            release=CapabilityReleaseIdentity.from_catalog_entry(forged_entry),
            recommendation_evidence=CapabilityRecommendationEvidence(
                recommendation_strategy_id="evil",
                reason_codes=(CapabilityRecommendationReasonCode.TOP_RANKED,),
            ),
            governance_evidence_refs=("evil.forge",),
        )
        return (*recommendations, forged)


def test_custom_machine_policy_proof() -> None:
    catalog, _service, _consumer = _mixed_fixture_service()
    service = _machine_service(catalog, _consumer, acquisition_policy=_ToolOnlyPolicy())
    response = service.acquire(
        _acquire_request(context=MarketplaceQueryContext(tenant_id="tenant-a")),
    )
    assert all(
        item.release.discovery.kind is CapabilityKind.TOOL for item in response.recommendations
    )


def test_evil_widen_policy_proof() -> None:
    catalog, _service, consumer = _mixed_fixture_service()
    service = _machine_service(catalog, consumer, acquisition_policy=_EvilWidenPolicy())
    with pytest.raises(MachineCapabilityAcquisitionPolicyError, match="widen"):
        service.acquire(_acquire_request())


def test_machine_acquisition_service_does_not_access_catalog_service_private_state() -> None:
    root = Path(importlib.import_module("intergrax.marketplace.acquisition").__path__[0])
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.attr, str) or not node.attr.startswith("_"):
                continue
            if isinstance(node.value, ast.Attribute) and node.value.attr == "catalog_service":
                raise AssertionError(
                    f"{path.name} accesses catalog_service private attribute {node.attr!r}",
                )


def test_acquisition_response_completeness_comes_from_same_pipeline_snapshot() -> None:
    ok_source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(_public_record(CapabilityKind.AGENT, "public-agent"),),
    )
    inner_catalog = FederatedCapabilityCatalog((ok_source,))
    federated = inner_catalog.snapshot()
    partial_snapshot = federated.model_copy(
        update={
            "federation_completeness": CapabilityCatalogFederationCompleteness.PARTIAL,
        },
    )
    complete_empty = CapabilityCatalogSnapshot(
        source_ids=federated.source_ids,
        entries=(),
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
    )

    class _FlippingSnapshotCatalog:
        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.sources = inner_catalog.sources

        def snapshot(self, **_kwargs: object) -> CapabilityCatalogSnapshot:
            self.snapshot_calls += 1
            if self.snapshot_calls == 1:
                return partial_snapshot
            return complete_empty

    flipping = _FlippingSnapshotCatalog()
    catalog = MarketplaceCatalogService(catalog=flipping, marketplace_sources=(ok_source,))
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    response = service.acquire(_acquire_request())
    assert response.outcome is MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE
    assert response.catalog_federation_completeness is MachineCatalogFederationCompleteness.PARTIAL
    assert flipping.snapshot_calls == 1


def test_acquire_does_not_read_second_snapshot_for_completeness() -> None:
    ok_source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(_public_record(CapabilityKind.AGENT, "public-agent"),),
    )
    inner = FederatedCapabilityCatalog((ok_source,))

    class _CountingCatalog:
        def __init__(self, wrapped: FederatedCapabilityCatalog) -> None:
            self._wrapped = wrapped
            self.snapshot_calls = 0

        @property
        def sources(self):
            return self._wrapped.sources

        def snapshot(self, **_kwargs: object) -> CapabilityCatalogSnapshot:
            self.snapshot_calls += 1
            return self._wrapped.snapshot(**_kwargs)

    counting = _CountingCatalog(inner)
    catalog = MarketplaceCatalogService(catalog=counting, marketplace_sources=(ok_source,))
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    service.acquire(_acquire_request())
    assert counting.snapshot_calls == 1


def _acquire_request_without_observation(
    **kwargs: object,
) -> MachineCapabilityAcquisitionRequest:
    return MachineCapabilityAcquisitionRequest(
        request_id="req-no-obs",
        need=_need(),
        discovery_query=_discovery_query(),
        marketplace_query_context=kwargs.pop("context", None) or MarketplaceQueryContext(),
        recommendation_context=CapabilityRecommendationContext(top_n=10),
        **kwargs,
    )


def test_acquire_without_observation_can_select_and_handoff() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    request = _acquire_request_without_observation()
    response = service.acquire(request)
    assert response.outcome is MachineCapabilityAcquisitionOutcome.RECOMMENDATIONS_AVAILABLE
    release = response.recommendations[0].release
    handoff = service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=request,
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-no-obs",
                discovery_correlation_id=response.discovery_correlation_id,
                selected_release=release,
                selector_id="machine.client",
            ),
            handoff_id="handoff-no-obs",
        ),
    )
    assert handoff.delivery.disposition is CapabilityHandoffDeliveryDisposition.DELIVERED


def test_generated_acquisition_correlation_reaches_handoff() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    request = _acquire_request_without_observation()
    response = service.acquire(request)
    release = response.recommendations[0].release
    service.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=request,
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-generated-corr",
                discovery_correlation_id=response.discovery_correlation_id,
                selected_release=release,
                selector_id="machine.client",
            ),
            handoff_id="handoff-generated-corr",
        ),
    )
    assert consumer.envelopes[0].discovery_correlation_id == response.discovery_correlation_id


def test_revalidation_preserves_original_discovery_correlation() -> None:
    _catalog, service, consumer = _mixed_fixture_service()
    request = _acquire_request_without_observation()
    response = service.acquire(request)
    release = response.recommendations[0].release
    revalidated = service._acquire(
        request,
        operation_discovery_correlation_id=response.discovery_correlation_id,
    )
    assert revalidated.discovery_correlation_id == response.discovery_correlation_id


def test_explicit_observation_correlation_mismatch_is_rejected() -> None:
    _, service, _consumer = _mixed_fixture_service()
    with pytest.raises(MachineCapabilityAcquisitionSelectionError, match="observation"):
        service._acquire(
            _acquire_request(correlation_id="corr-A"),
            operation_discovery_correlation_id="corr-B",
        )


def test_machine_acquisition_handoff_request_has_unique_schema_version() -> None:
    _, service, _consumer = _mixed_fixture_service()
    acquire_req = _acquire_request(correlation_id="corr-schema")
    release = service.acquire(acquire_req).recommendations[0].release
    request = MachineCapabilityAcquisitionHandoffRequest(
        acquisition_request=acquire_req,
        selection=MachineCapabilityAcquisitionSelection(
            selection_id="sel-schema",
            discovery_correlation_id="corr-schema",
            selected_release=release,
            selector_id="machine.client",
        ),
        handoff_id="handoff-schema",
    )
    assert request.schema_version == SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1
    assert request.schema_version != SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1


def test_me12_public_contract_schema_versions_are_unique() -> None:
    schema_ids = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
        SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    )
    assert len(schema_ids) == len(set(schema_ids))


def test_me12_acquisition_contract_serialization_roundtrip() -> None:
    acquire_req = _acquire_request(correlation_id="corr-rt")
    _, service, _consumer = _mixed_fixture_service()
    acquire_resp = service.acquire(acquire_req)
    release = acquire_resp.recommendations[0].release
    handoff_req = MachineCapabilityAcquisitionHandoffRequest(
        acquisition_request=acquire_req,
        selection=MachineCapabilityAcquisitionSelection(
            selection_id="sel-rt",
            discovery_correlation_id="corr-rt",
            selected_release=release,
            selector_id="machine.client",
        ),
        handoff_id="handoff-rt",
    )
    handoff_resp = MachineCapabilityAcquisitionHandoffResponse(
        request_id=acquire_req.request_id,
        discovery_correlation_id="corr-rt",
        selection_id="sel-rt",
        handoff_id="handoff-rt",
        delivery=CapabilityHandoffDeliveryResult(
            disposition=CapabilityHandoffDeliveryDisposition.DELIVERED,
            handoff_id="handoff-rt",
            downstream_consumer_id="machine.consumer.me12",
        ),
    )
    for model in (acquire_req, acquire_resp, handoff_req, handoff_resp):
        payload = model.model_dump_json()
        restored = type(model).model_validate_json(payload)
        assert restored.schema_version == model.schema_version


def test_machine_partial_catalog_reports_completeness() -> None:
    class _FailingSource:
        @property
        def source_id(self) -> str:
            return "zzz.failing"

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            raise CapabilityCatalogSourceFailure("unavailable")

    class _FixedPartialSnapshotCatalog:
        def __init__(self, snapshot, sources: tuple[object, ...]) -> None:
            self._snapshot = snapshot
            self.sources = sources

        def snapshot(self, **_kwargs: object):
            return self._snapshot

    ok_source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(_public_record(CapabilityKind.AGENT, "public-agent"),),
    )
    federated = FederatedCapabilityCatalog((ok_source, _FailingSource())).snapshot(
        federation_policy=CapabilityCatalogFederationPolicy.ALLOW_PARTIAL,
    )
    service_catalog = MarketplaceCatalogService(
        catalog=_FixedPartialSnapshotCatalog(federated, (ok_source,)),
        marketplace_sources=(ok_source,),
    )
    consumer = _RecordingHandoffConsumer()
    service = _machine_service(
        service_catalog,
        consumer,
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
    )
    response = service.acquire(_acquire_request())
    assert response.catalog_federation_completeness is MachineCatalogFederationCompleteness.PARTIAL
    assert federated.federation_completeness.value == "partial"


def test_me12_acquisition_contracts_do_not_import_worker_engine() -> None:
    path = Path(importlib.import_module("intergrax.contracts.marketplace.acquisition").__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = (
        "intergrax.autonomous_work",
        "intergrax.contracts.autonomous_work",
        "intergrax.agent_distribution",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden:
                if node.module == prefix or node.module.startswith(f"{prefix}."):
                    raise AssertionError(f"forbidden import {node.module}")


def test_me12_acquisition_implementation_has_no_nexus_imports() -> None:
    root = Path(importlib.import_module("intergrax.marketplace.acquisition").__path__[0])
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("intergrax.nexus"):
                    raise AssertionError(f"{path.name} imports nexus")


def test_me12_no_duplicate_machine_marketplace_engine_symbols() -> None:
    root = Path(importlib.import_module("intergrax.marketplace.acquisition").__path__[0])
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "MachineMarketplaceEngine" not in text
        assert "WorkerMarketplaceEngine" not in text
