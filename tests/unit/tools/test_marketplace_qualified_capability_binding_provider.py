# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P2 marketplace tool binding provider tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    QualifiedCapabilitySubject,
    QualifiedCapabilitySubjectKind,
    derive_qualified_subject_reference,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
    MarketplaceQualifiedToolStageIntegrityError,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MarketplaceToolQualifiedCapabilityBindingProvider,
    execution_target_reference_for_marketplace_qualified_tool,
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
_STRATEGY = "marketplace.gap_acquisition.v1"
_TASK_ID = TaskId("task_00000000000000000000000000000001")
_SOURCE = CapabilitySourceIdentity(
    source_id="official.gap02.bind",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


def _release() -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.bind",
            ),
        ),
        publisher="publisher",
        version_label="1.0.0",
        content_digest="sha256:bind",
        package_reference="pkg://bind",
    )


def _binding_stack(
    *,
    tenant_id: str = "tenant-1",
    acquisition_id: str = "acq-bind",
) -> tuple[MarketplaceToolQualifiedCapabilityBindingProvider, str, str]:
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    provider = MarketplaceToolQualifiedCapabilityBindingProvider(
        stage_repository=stage_repo,
        context_resolver=resolver,
    )
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=tenant_id,
        operation_id=acquisition_id,
    )
    assoc_repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id=tenant_id,
            acquisition_request_id=acquisition_id,
        ),
    )
    stage_repo.stage(
        MarketplaceQualifiedToolStage(
            handoff_id=handoff_id,
            tenant_id=tenant_id,
            selected_release=_release(),
            discovery_correlation_id="discovery-1",
            selection_id=f"marketplace-gap-selection:{acquisition_id}",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=_NOW,
        ),
    )
    return provider, acquisition_id, marketplace_domain_handoff_reference(handoff_id)


def _subject(domain_ref: str, qualification_request_id: str = "qual-req-1") -> QualifiedCapabilitySubject:
    return QualifiedCapabilitySubject(
        qualification_request_id=qualification_request_id,
        subject_kind=QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE,
        subject_reference=domain_ref,
        qualified_subject_reference=derive_qualified_subject_reference(
            qualification_request_id=qualification_request_id,
            subject_kind=QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE,
            subject_reference=domain_ref,
        ),
    )


def _qualification(
    *,
    acquisition_id: str,
    domain_ref: str,
    qualification_request_id: str = "qual-req-1",
) -> CapabilityQualificationResult:
    return CapabilityQualificationResult(
        qualification_request_id=qualification_request_id,
        acquisition_request_id=acquisition_id,
        gap_id="gap-1",
        strategy_id=_STRATEGY,
        provider_id="marketplace.tool.qualification.v1",
        outcome=CapabilityQualificationOutcome.QUALIFIED,
        reason_code=CapabilityQualificationReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityQualificationEvidence(
            provider_id="marketplace.tool.qualification.v1",
            qualification_request_id=qualification_request_id,
            acquisition_request_id=acquisition_id,
            acquisition_strategy_id=_STRATEGY,
            gap_id="gap-1",
            domain_handoff_reference=domain_ref,
        ),
    )


def test_qualified_domain_handoff_bound() -> None:
    provider, acquisition_id, domain_ref = _binding_stack()
    subject = _subject(domain_ref)
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id="resume-1",
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id="resume-1",
        qualified_subject=subject,
        qualification_result=_qualification(
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
        worker_need_id="worker-need-1",
        worker_instance_id="worker-1",
        tenant_id="tenant-1",
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.BOUND
    assert result.execution_target is not None
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-1",
        operation_id=acquisition_id,
    )
    assert (
        result.execution_target.execution_target_reference
        == execution_target_reference_for_marketplace_qualified_tool(handoff_id)
    )


def test_tenant_mismatch_subject_mismatch() -> None:
    provider, acquisition_id, domain_ref = _binding_stack(tenant_id="tenant-1")
    subject = _subject(domain_ref)
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id="resume-1",
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id="resume-1",
        qualified_subject=subject,
        qualification_result=_qualification(
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
        worker_need_id="worker-need-1",
        worker_instance_id="worker-1",
        tenant_id="tenant-other",
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.CONFLICT
    assert result.reason_code is QualifiedCapabilityBindingReasonCode.SUBJECT_MISMATCH


def test_stage_integrity_maps_to_integrity_conflict() -> None:
    store = InMemoryDocumentStore()
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    acquisition_id = "acq-bind-integrity"
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id="tenant-1",
        operation_id=acquisition_id,
    )
    assoc_repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id="tenant-1",
            acquisition_request_id=acquisition_id,
        ),
    )
    domain_ref = marketplace_domain_handoff_reference(handoff_id)

    class _IntegrityStageRepo:
        def stage(self, record: MarketplaceQualifiedToolStage):
            raise NotImplementedError

        def get(
            self,
            *,
            tenant_id: str,
            handoff_id: str,
        ) -> MarketplaceQualifiedToolStage | None:
            raise MarketplaceQualifiedToolStageIntegrityError("corrupt stage")

    provider = MarketplaceToolQualifiedCapabilityBindingProvider(
        stage_repository=_IntegrityStageRepo(),
        context_resolver=resolver,
    )
    subject = _subject(domain_ref)
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id="resume-1",
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id="resume-1",
        qualified_subject=subject,
        qualification_result=_qualification(
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
        worker_need_id="worker-need-1",
        worker_instance_id="worker-1",
        tenant_id="tenant-1",
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    result = provider.bind(request)
    assert result.outcome is QualifiedCapabilityBindingOutcome.CONFLICT
    assert result.reason_code is QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT


def test_repeated_identical_binding_same_target() -> None:
    provider, acquisition_id, domain_ref = _binding_stack()
    subject = _subject(domain_ref)
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id="resume-1",
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    request = QualifiedCapabilityBindingRequest(
        binding_operation_id=binding_id,
        resume_operation_id="resume-1",
        qualified_subject=subject,
        qualification_result=_qualification(
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
        worker_need_id="worker-need-1",
        worker_instance_id="worker-1",
        tenant_id="tenant-1",
        task_id=_TASK_ID,
        requested_at=_NOW,
    )
    first = provider.bind(request)
    second = provider.bind(request)
    assert first.execution_target == second.execution_target


def test_no_activation_dependencies() -> None:
    path = Path("intergrax/tools/marketplace_qualified_capability_binding_provider.py")
    source = path.read_text(encoding="utf-8")
    for token in ("ToolRegistry", "ToolRuntime", "DynamicToolAcquisition", "ExecutionEngine"):
        assert token not in source
