# © Artur Czarnecki. All rights reserved.

"""UCA-5 — gap acquisition service single-pass handoff."""

from __future__ import annotations

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
    CapabilityKind,
    CapabilityNeed,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.marketplace import (
    MarketplaceListingRecord,
    MarketplaceQueryContext,
)
from intergrax.contracts.marketplace.acquisition import (
    MachineCapabilityAcquisitionHandoffRequest,
    MachineCapabilityAcquisitionRequest,
    MachineCapabilityAcquisitionSelection,
)
from intergrax.contracts.capability_catalog.recommendation import (
    CapabilityRecommendationContext,
)
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionRequest,
)
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import (
    MachineCapabilityAcquisitionService,
    MarketplaceGapAcquisitionService,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)

pytestmark = pytest.mark.unit

_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.intergrax.uca5",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _catalog_service(*records: MarketplaceListingRecord) -> MarketplaceCatalogService:
    source = MarketplaceCapabilityCatalogSource(source=_OFFICIAL, records=records)
    catalog = FederatedCapabilityCatalog((source,))
    return MarketplaceCatalogService(catalog=catalog, marketplace_sources=(source,))


class _RecordingHandoffConsumer:
    def __init__(self) -> None:
        self.envelopes: list = []

    @property
    def consumer_id(self) -> str:
        return "uca5.test.consumer"

    def consume(self, envelope) -> None:
        self.envelopes.append(envelope)


def _build_services():
    catalog = _catalog_service(
        MarketplaceListingRecord(
            kind=CapabilityKind.TOOL,
            logical_id="tool.uca5",
            display_label="tool.uca5",
            publisher="intergrax",
        ),
    )
    consumer = _RecordingHandoffConsumer()
    trace = InMemoryCapabilityHandoffTraceEvidenceConsumer()
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=trace,
    )
    orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=catalog,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        delivery_service=delivery,
    )
    gap_service = MarketplaceGapAcquisitionService(
        catalog_service=catalog,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
        handoff_orchestrator=orchestrator,
    )
    machine = MachineCapabilityAcquisitionService(
        catalog_service=catalog,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
        handoff_orchestrator=orchestrator,
    )
    return gap_service, machine, consumer


def test_gap_service_handoff_success_produces_domain_reference() -> None:
    gap_service, _machine, _consumer = _build_services()
    request = MarketplaceGapAcquisitionRequest(
        operation_id="op-uca5-success",
        gap_id="capability-gap:need-1:canonical-corr",
        canonical_discovery_correlation_id="canonical-corr",
        capability_need=CapabilityNeed(
            need_id="need-1",
            kinds=(CapabilityKind.TOOL,),
            intent_summary="tool",
        ),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
    )
    result = gap_service.acquire_from_gap(request)
    assert result.outcome is MarketplaceGapAcquisitionOutcome.SUCCEEDED
    assert result.domain_handoff_reference is not None
    assert result.domain_handoff_reference.startswith("handoff://")


def test_select_and_handoff_runs_listing_pipeline_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gap_service, machine, _consumer = _build_services()
    calls = {"count": 0}
    from intergrax.marketplace.observed_pipeline import (
        run_marketplace_intelligence_pipeline as original_pipeline,
    )

    def counting_pipeline(**kwargs: object):
        calls["count"] += 1
        return original_pipeline(**kwargs)

    monkeypatch.setattr(
        "intergrax.marketplace.acquisition.service.run_marketplace_intelligence_pipeline",
        counting_pipeline,
    )
    acquire_request = MachineCapabilityAcquisitionRequest(
        request_id="req-handoff-once",
        need=CapabilityNeed(kinds=(CapabilityKind.TOOL,)),
        discovery_query=_discovery_query(),
        marketplace_query_context=MarketplaceQueryContext(),
        recommendation_context=CapabilityRecommendationContext(top_n=10),
    )
    acquire = machine.acquire(acquire_request)
    release = acquire.recommendations[0].release
    machine.select_and_handoff(
        MachineCapabilityAcquisitionHandoffRequest(
            acquisition_request=acquire_request,
            selection=MachineCapabilityAcquisitionSelection(
                selection_id="sel-once",
                discovery_correlation_id=acquire.discovery_correlation_id,
                selected_release=release,
                selector_id="test",
            ),
            handoff_id="handoff-once",
        ),
    )
    assert calls["count"] == 2
