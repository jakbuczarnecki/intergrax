# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 bounded integration: acquire → stage → qualify → bind."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.capability_catalog import (
    AvailabilityPreservingGovernanceEvaluator,
    FederatedCapabilityCatalog,
)
from intergrax.capability_qualification.qualification_service import (
    CapabilityQualificationService,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
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
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubjectKind,
    qualified_capability_subject_from_result,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.marketplace import MarketplaceListingRecord, MarketplaceQueryContext
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import (
    MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
    MarketplaceGapAcquisitionService,
    MarketplaceGapCapabilityAcquisitionStrategy,
)
from intergrax.marketplace.handoff.adapters.tool_qualification_staging_consumer import (
    ToolQualificationStagingConsumer,
)
from intergrax.marketplace.handoff_traceability import (
    CapabilityHandoffDeliveryService,
    InMemoryCapabilityHandoffDeliveryAdmission,
    InMemoryCapabilityHandoffTraceEvidenceConsumer,
    MarketplaceDiscoveryHandoffOrchestrator,
)
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MarketplaceToolQualifiedCapabilityBindingProvider,
)
from intergrax.tools.marketplace_qualified_capability_qualification_provider import (
    MarketplaceToolCapabilityQualificationProvider,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_recorder import (
    MarketplaceQualifiedToolStageContextRecorderImpl,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_resolver import (
    MarketplaceQualifiedToolStageContextResolverImpl,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 3, 26, 12, 0, tzinfo=UTC)
_OFFICIAL = CapabilitySourceIdentity(
    source_id="official.gap02.integration",
    source_kind=CapabilitySourceKind.OFFICIAL,
)
_TASK_ID = TaskId("task_00000000000000000000000000000002")


def _discovery_query() -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(
        scope=CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL),
    )


def _capability_gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _gap_service(
    tenant_id: str,
) -> tuple[MarketplaceGapAcquisitionService, InMemoryDocumentStore]:
    source = MarketplaceCapabilityCatalogSource(
        source=_OFFICIAL,
        records=(
            MarketplaceListingRecord(
                kind=CapabilityKind.TOOL,
                logical_id="tool.gap02",
                display_label="tool.gap02",
                publisher="intergrax",
            ),
        ),
    )
    catalog = FederatedCapabilityCatalog((source,))
    catalog_service = MarketplaceCatalogService(
        catalog=catalog,
        marketplace_sources=(source,),
    )
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    recorder = MarketplaceQualifiedToolStageContextRecorderImpl(assoc_repo)
    consumer = ToolQualificationStagingConsumer(stage_repo, assoc_repo)
    delivery = CapabilityHandoffDeliveryService(
        consumer=consumer,
        delivery_admission=InMemoryCapabilityHandoffDeliveryAdmission(),
        trace_evidence_consumer=InMemoryCapabilityHandoffTraceEvidenceConsumer(),
    )
    orchestrator = MarketplaceDiscoveryHandoffOrchestrator(
        catalog_service=catalog_service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        delivery_service=delivery,
    )
    gap_service = MarketplaceGapAcquisitionService(
        catalog_service=catalog_service,
        discovery_service=MarketplaceDiscoveryService.with_defaults(),
        governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
        governance_context=CapabilityGovernanceContext(),
        recommendation_service=MarketplaceRecommendationService.with_defaults(),
        handoff_orchestrator=orchestrator,
        tool_stage_context_recorder=recorder,
    )
    return gap_service, store


def _run_tenant_flow(tenant_id: str, *, request_nonce: str) -> None:
    gap_service, store = _gap_service(tenant_id)
    strategy = MarketplaceGapCapabilityAcquisitionStrategy(
        gap_service,
        marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
    )
    gap = _capability_gap()
    need = CapabilityNeed(
        need_id="need-1",
        kinds=(CapabilityKind.TOOL,),
        intent_summary="tool",
    )
    acquisition_id = derive_capability_acquisition_request_id(
        gap_id=gap.gap_id,
        request_nonce=request_nonce,
    )
    acq_request = CapabilityAcquisitionRequest(
        request_id=acquisition_id,
        request_nonce=request_nonce,
        capability_gap=gap,
        capability_need=need,
        requested_at=_NOW,
    )
    acquisition = strategy.acquire(acq_request)
    assert acquisition.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    assert acquisition.strategy_id == MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
    assert acquisition.evidence is not None
    assert acquisition.evidence.domain_handoff_reference is not None
    assert acquisition.evidence.evidence_ref is not None
    assert acquisition.evidence.artifact_reference is None

    qual_request = CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=acquisition_id,
            qualification_nonce=f"nonce-{tenant_id}",
        ),
        qualification_nonce=f"nonce-{tenant_id}",
        acquisition_request_id=acquisition_id,
        gap_id=acquisition.gap_id,
        strategy_id=MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
        acquisition_result=acquisition,
        requested_at=_NOW,
    )
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    qualification_service = CapabilityQualificationService(
        (
            MarketplaceToolCapabilityQualificationProvider(
                stage_repository=stage_repo,
                context_resolver=resolver,
            ),
        ),
    )
    qual_decision = qualification_service.qualify(qual_request)
    assert (
        qual_decision.qualification_result.outcome
        is CapabilityQualificationOutcome.QUALIFIED
    )
    subject = qualified_capability_subject_from_result(
        qual_decision.qualification_result,
    )
    assert subject is not None
    assert subject.subject_kind is QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE

    binding_service = QualifiedCapabilityBindingService(
        (
            MarketplaceToolQualifiedCapabilityBindingProvider(
                stage_repository=stage_repo,
                context_resolver=resolver,
            ),
        ),
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=f"resume-{tenant_id}",
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    binding_result = binding_service.bind(
        QualifiedCapabilityBindingRequest(
            binding_operation_id=binding_id,
            resume_operation_id=f"resume-{tenant_id}",
            qualified_subject=subject,
            qualification_result=qual_decision.qualification_result,
            worker_need_id="worker-need-1",
            worker_instance_id="worker-1",
            tenant_id=tenant_id,
            task_id=_TASK_ID,
            requested_at=_NOW,
        ),
    )
    assert binding_result.outcome is QualifiedCapabilityBindingOutcome.BOUND


def test_p2_integration_single_tenant() -> None:
    _run_tenant_flow("tenant-a", request_nonce="nonce-integration-1")


def test_multi_tenant_same_acquisition_id() -> None:
    shared_nonce = "shared-acquisition-nonce"
    _run_tenant_flow("tenant-a", request_nonce=shared_nonce)
    _run_tenant_flow("tenant-b", request_nonce=shared_nonce)


def test_integration_modules_have_no_execution_activation() -> None:
    tokens = ("ToolRegistry", "ToolRuntime", "DynamicToolAcquisition", "ExecutionEngine")
    for relative in (
        "intergrax/tools/marketplace_qualified_capability_qualification_provider.py",
        "intergrax/tools/marketplace_qualified_capability_binding_provider.py",
        "intergrax/marketplace/handoff/adapters/tool_qualification_staging_consumer.py",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        for token in tokens:
            assert token not in source
