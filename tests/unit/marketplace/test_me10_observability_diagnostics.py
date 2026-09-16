# © Artur Czarnecki. All rights reserved.

"""ME-10 pluggable marketplace observability diagnostics qualification."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityKind,
    CapabilityIdentityKey,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    MarketplaceListingRecord,
    MarketplaceQueryContext,
    MarketplacePipelineStage,
    MarketplaceObserverFailurePolicy,
    MarketplaceObserverEmitError,
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.contracts.marketplace.diagnostics import MarketplaceDiagnosticEvent
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.diagnostics import (
    InMemoryMarketplaceDiagnosticObserver,
    MarketplacePipelineObservationSession,
    NoOpMarketplaceDiagnosticObserver,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    MarketplaceDiscoveryHandoffOrchestrator,
)
from intergrax.marketplace.observed_pipeline import run_marketplace_intelligence_pipeline
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.me10obs",
    source_kind=CapabilitySourceKind.OFFICIAL,
)

_FORBIDDEN_TELEMETRY_PREFIXES = (
    "opentelemetry",
    "opentelemetry.sdk",
    "datadog",
    "newrelic",
    "prometheus_client",
    "splunk",
)


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _record(
    kind: CapabilityKind,
    logical_id: str,
    *,
    tenant_id: str | None = None,
) -> MarketplaceListingRecord:
    visibility = None
    if tenant_id is not None:
        visibility = MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id=tenant_id,
        )
    return MarketplaceListingRecord(
        kind=kind,
        logical_id=logical_id,
        display_label=logical_id,
        publisher="intergrax",
        version_label="1.0.0",
        content_digest=f"sha256:{logical_id}",
        visibility=visibility,
    )


def _catalog_service(*records: MarketplaceListingRecord) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _CustomMarketplaceDiagnosticSink:
    """Structural plugin — no subclass of platform defaults."""

    def __init__(self) -> None:
        self.events: list[MarketplaceDiagnosticEvent] = []

    @property
    def observer_id(self) -> str:
        return "custom.me10.diagnostic_sink"

    def emit(self, event: MarketplaceDiagnosticEvent) -> None:
        self.events.append(event)


class _FailingObserver:
    @property
    def observer_id(self) -> str:
        return "custom.me10.failing"

    def emit(self, event: MarketplaceDiagnosticEvent) -> None:
        raise RuntimeError("observer defect")


def _run_pipeline(
    service: MarketplaceCatalogService,
    observer: InMemoryMarketplaceDiagnosticObserver | None,
    *,
    tenant_id: str | None = None,
) -> tuple:
    observation = None
    if observer is not None:
        observation = MarketplacePipelineObservationSession.for_discovery(
            "corr-me10-pipeline",
            observer=observer,
        )
    return run_marketplace_intelligence_pipeline(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
        observation=observation,
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
    )


def test_marketplace_observer_is_pluginable_without_core_changes() -> None:
    sink = _CustomMarketplaceDiagnosticSink()
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.plugin"))
    _run_pipeline(service, None)
    observation = MarketplacePipelineObservationSession.for_discovery(
        "corr-plugin",
        observer=sink,
    )
    run_marketplace_intelligence_pipeline(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        observation=observation,
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
    )
    assert sink.events
    assert sink.observer_id == "custom.me10.diagnostic_sink"


def test_observability_does_not_change_business_result() -> None:
    service = _catalog_service(
        _record(CapabilityKind.TOOL, "tools.obs.alpha"),
        _record(CapabilityKind.SKILL, "skills.obs.beta"),
    )
    without = _run_pipeline(service, None)
    with_obs = _run_pipeline(
        service,
        InMemoryMarketplaceDiagnosticObserver(),
    )
    assert without.recommendations == with_obs.recommendations
    assert without.governed.allowed == with_obs.governed.allowed
    assert without.ranked == with_obs.ranked
    assert without.listing_views == with_obs.listing_views


def test_discovery_visibility_search_rank_govern_recommend_trace_is_correlated() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.trace"))
    _run_pipeline(service, observer)
    corr = "corr-me10-pipeline"
    for event in observer.events:
        assert event.correlation.discovery_correlation_id == corr
    stages = {event.stage for event in observer.events}
    assert MarketplacePipelineStage.VISIBILITY in stages
    assert MarketplacePipelineStage.SEARCH in stages
    assert MarketplacePipelineStage.RANKING in stages
    assert MarketplacePipelineStage.GOVERNANCE in stages
    assert MarketplacePipelineStage.RECOMMENDATION in stages


def test_visibility_diagnostics_do_not_leak_foreign_private_capability_identity() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(
        _record(CapabilityKind.AGENT, "agents.secret.foreign", tenant_id="tenant-b"),
        _record(CapabilityKind.AGENT, "agents.public.local"),
    )
    _run_pipeline(service, observer, tenant_id="tenant-a")
    visibility_events = [
        e for e in observer.events if e.stage is MarketplacePipelineStage.VISIBILITY
    ]
    assert visibility_events
    serialized = " ".join(str(e.model_dump()) for e in visibility_events)
    assert "agents.secret.foreign" not in serialized
    assert "tenant-b" not in serialized
    assert visibility_events[0].filtered_count == 1


def test_search_diagnostics_include_strategy_and_counts() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(_record(CapabilityKind.TOOL, "tools.obs.search"))
    _run_pipeline(service, observer)
    search_events = [e for e in observer.events if e.stage is MarketplacePipelineStage.SEARCH]
    assert search_events
    assert any(e.strategy_id and e.input_count is not None and e.output_count is not None for e in search_events)


def test_ranking_diagnostics_include_ranker_and_counts() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(_record(CapabilityKind.SKILL, "skills.obs.rank"))
    _run_pipeline(service, observer)
    rank_events = [e for e in observer.events if e.stage is MarketplacePipelineStage.RANKING]
    assert rank_events
    event = rank_events[0]
    assert event.ranker_id
    assert event.input_count is not None
    assert event.output_count is not None


def test_governance_diagnostics_reference_existing_evidence() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.gov"))
    _run_pipeline(service, observer)
    gov = next(e for e in observer.events if e.stage is MarketplacePipelineStage.GOVERNANCE)
    assert gov.governance_evaluator_ids
    assert gov.governance_evidence_refs
    assert gov.allowed_count is not None
    assert gov.blocked_count is not None


def test_recommendation_diagnostics_include_strategy_and_counts() -> None:
    observer = InMemoryMarketplaceDiagnosticObserver()
    service = _catalog_service(_record(CapabilityKind.TOOL, "tools.obs.rec"))
    _run_pipeline(service, observer)
    rec = next(e for e in observer.events if e.stage is MarketplacePipelineStage.RECOMMENDATION)
    assert rec.recommendation_strategy_id
    assert rec.input_count is not None
    assert rec.output_count is not None


def test_lifecycle_handoff_diagnostics_preserve_capability_identity_and_disposition() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.handoff"))
    observer = InMemoryMarketplaceDiagnosticObserver()

    class _Consumer:
        @property
        def consumer_id(self) -> str:
            return "consumer.me10"

        def consume(self, envelope) -> None:
            return None

    delivery = CapabilityHandoffDeliveryService(consumer=_Consumer())
    orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        delivery_service=delivery,
        diagnostic_observer=observer,
    )
    entry = service._catalog.snapshot().entries[0]
    key = CapabilityIdentityKey.from_discovery_identity(entry.identity)
    orchestrator.execute_explicit_selection_handoff(
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        selected_identity_key=key,
        consumer_target=CapabilityHandoffConsumerTarget.AGENT_DOMAIN,
        selector_id="operator",
        discovery_correlation_id="corr-handoff",
        selection_id="sel-1",
        handoff_id="ho-1",
        downstream_consumer_id="consumer.me10",
    )
    handoff_events = [e for e in observer.events if e.stage is MarketplacePipelineStage.HANDOFF]
    assert handoff_events
    completed = [e for e in handoff_events if e.handoff_status]
    assert completed
    assert completed[-1].selected_release is not None


def test_custom_observer_failure_follows_explicit_policy() -> None:
    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.fail"))
    observation = MarketplacePipelineObservationSession.for_discovery(
        "corr-fail",
        observer=_FailingObserver(),
        failure_policy=MarketplaceObserverFailurePolicy.STRICT,
    )
    with pytest.raises(MarketplaceObserverEmitError):
        run_marketplace_intelligence_pipeline(
            catalog_service=service,
            discovery_service=MarketplaceDiscoveryService.with_defaults(),
            governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            governance_context=CapabilityGovernanceContext(
                posture=CapabilityGovernancePosture.STRICT,
            ),
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(),
            observation=observation,
            recommendation_service=MarketplaceRecommendationService.with_defaults(),
        )

    best_effort = MarketplacePipelineObservationSession.for_discovery(
        "corr-fail-be",
        observer=_FailingObserver(),
        failure_policy=MarketplaceObserverFailurePolicy.BEST_EFFORT,
    )
    result = run_marketplace_intelligence_pipeline(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        observation=best_effort,
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
    )
    assert result.listing_views


def test_unexpected_programming_error_is_not_normalized_by_observability() -> None:
    class _BrokenRanker:
        @property
        def ranker_id(self) -> str:
            return "broken.ranker"

        def rank(self, candidates, context):
            raise RuntimeError("programming defect")

    service = _catalog_service(_record(CapabilityKind.AGENT, "agents.obs.broken"))
    discovery = MarketplaceDiscoveryService(
        search_strategy=MarketplaceDiscoveryService.with_defaults().search_strategy,
        ranker=_BrokenRanker(),
    )
    observation = MarketplacePipelineObservationSession.for_discovery(
        "corr-broken",
        observer=InMemoryMarketplaceDiagnosticObserver(),
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        run_marketplace_intelligence_pipeline(
            catalog_service=service,
            discovery_service=discovery,
            governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            governance_context=CapabilityGovernanceContext(
                posture=CapabilityGovernancePosture.STRICT,
            ),
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(),
            observation=observation,
            recommendation_service=MarketplaceRecommendationService.with_defaults(),
        )


def test_marketplace_core_has_no_telemetry_vendor_dependencies() -> None:
    package = importlib.import_module("intergrax.marketplace")
    root = Path(package.__path__[0])
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for imported in modules:
                for prefix in _FORBIDDEN_TELEMETRY_PREFIXES:
                    if imported == prefix or imported.startswith(f"{prefix}."):
                        raise AssertionError(
                            f"{path.relative_to(root)} imports forbidden telemetry: {imported}",
                        )


def test_noop_observer_matches_absent_observer_business_result() -> None:
    service = _catalog_service(_record(CapabilityKind.SKILL, "skills.noop"))
    absent = _run_pipeline(service, None)
    noop_session = MarketplacePipelineObservationSession.for_discovery(
        "corr-noop",
        observer=NoOpMarketplaceDiagnosticObserver(),
    )
    with_noop = run_marketplace_intelligence_pipeline(
        catalog_service=service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(
            posture=CapabilityGovernancePosture.STRICT,
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        observation=noop_session,
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
    )
    assert absent.listing_views == with_noop.listing_views
    assert absent.recommendations == with_noop.recommendations
