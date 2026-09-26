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
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
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
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionOutcome,
    MarketplaceGapAcquisitionRequest,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.marketplace import (
    MarketplaceCapabilityCatalogSource,
    MarketplaceCatalogService,
    MarketplaceDiscoveryService,
    MarketplaceRecommendationService,
)
from intergrax.marketplace.acquisition import MarketplaceGapAcquisitionService
from intergrax.marketplace.acquisition.uca_acquisition_strategy import (
    MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
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
    )
    return gap_service, store


def _run_tenant_flow(tenant_id: str, acquisition_id: str) -> None:
    gap_service, store = _gap_service(tenant_id)
    gap_result = gap_service.acquire_from_gap(
        MarketplaceGapAcquisitionRequest(
            operation_id=acquisition_id,
            gap_id="capability-gap:need-1:corr",
            canonical_discovery_correlation_id="corr",
            capability_need=CapabilityNeed(
                need_id="need-1",
                kinds=(CapabilityKind.TOOL,),
                intent_summary="tool",
            ),
            discovery_query=_discovery_query(),
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
        ),
    )
    assert gap_result.outcome is MarketplaceGapAcquisitionOutcome.SUCCEEDED
    assert gap_result.domain_handoff_reference is not None

    acquisition = CapabilityAcquisitionResult(
        request_id=acquisition_id,
        gap_id="capability-gap:need-1:corr",
        strategy_id=MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID,
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityAcquisitionEvidence(
            domain_handoff_reference=gap_result.domain_handoff_reference,
        ),
    )
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
    _run_tenant_flow("tenant-a", "acq-integration-1")


def test_multi_tenant_same_acquisition_id() -> None:
    acquisition_id = "shared-acquisition-id"
    _run_tenant_flow("tenant-a", acquisition_id)
    _run_tenant_flow("tenant-b", acquisition_id)


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
