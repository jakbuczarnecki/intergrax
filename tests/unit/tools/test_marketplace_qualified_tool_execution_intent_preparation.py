# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
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
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.marketplace_qualified_capability import (
    MarketplaceQualifiedToolStage,
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
)
from intergrax.contracts.tools.qualified_capability_execution_intent_preparation import (
    QualifiedCapabilityExecutionIntentPreparationOutcome,
    QualifiedCapabilityExecutionIntentPreparationRequest,
)
from intergrax.contracts.tools.qualified_marketplace_tool_execution_intent import (
    QualifiedMarketplaceToolExecutionIntentConflictError,
    QualifiedMarketplaceToolExecutionIntentUnavailableError,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.marketplace.acquisition import MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
from intergrax.runtime.codecraft.acquisition import CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.marketplace_qualified_tool_execution_intent_preparation import (
    MarketplaceQualifiedToolExecutionIntentPreparation,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_association import (
    DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository,
)
from intergrax.tools.marketplace_qualified_tool_stage_context_resolver import (
    MarketplaceQualifiedToolStageContextResolverImpl,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _WORKER_ID,
)
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import (
    _NOW,
    _STRATEGY,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_00000000000000000000000000000001")


def _release() -> CapabilityReleaseIdentity:
    source = CapabilitySourceIdentity(
        source_id="official.gap02.prep",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tools.prep",
            ),
        ),
        publisher="publisher",
        version_label="1.0.0",
        content_digest="sha256:prep",
        package_reference="pkg://prep",
    )


def _stack(
    *,
    tenant_id: str = "tenant-1",
    acquisition_id: str = "acq-prep",
) -> tuple[
    MarketplaceQualifiedToolExecutionIntentPreparation,
    str,
    str,
    DocumentStoreMarketplaceQualifiedToolStageRepository,
    MarketplaceQualifiedToolStageContextResolverImpl,
]:
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
        intent_repository=intent_repo,
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
    return (
        preparation,
        acquisition_id,
        marketplace_domain_handoff_reference(handoff_id),
        stage_repo,
        resolver,
    )


def _need() -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:prep",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id="recovery:prep",
    )


def _qualification(acquisition_id: str, domain_ref: str) -> CapabilityQualificationResult:
    qid = "qual-req-prep"
    return CapabilityQualificationResult(
        qualification_request_id=qid,
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
            qualification_request_id=qid,
            acquisition_request_id=acquisition_id,
            acquisition_strategy_id=_STRATEGY,
            gap_id="gap-1",
            domain_handoff_reference=domain_ref,
        ),
    )


def _request(
    *,
    preparation: MarketplaceQualifiedToolExecutionIntentPreparation,
    acquisition_id: str,
    domain_ref: str,
    tenant_id: str = "tenant-1",
) -> QualifiedCapabilityExecutionIntentPreparationRequest:
    qualification = _qualification(acquisition_id, domain_ref)
    subject_ref = derive_qualified_subject_reference(
        qualification_request_id=qualification.qualification_request_id,
        subject_kind=QualifiedCapabilitySubjectKind.DOMAIN_HANDOFF_REFERENCE,
        subject_reference=domain_ref,
    )
    resume_id = "resume-prep"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject_ref,
    )
    from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
        derive_qualified_capability_execution_request_id,
    )

    execution_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    return QualifiedCapabilityExecutionIntentPreparationRequest(
        need=_need(),
        qualification_result=qualification,
        execution_request_id=execution_id,
        binding_operation_id=binding_id,
        resume_operation_id=resume_id,
        worker_need_id="worker-need-prep",
        tenant_id=tenant_id,
        task_id=_TASK_ID,
    )


def test_marketplace_created() -> None:
    preparation, acquisition_id, domain_ref, _, _ = _stack()
    result = preparation.prepare(
        _request(
            preparation=preparation,
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
    )
    assert result.outcome is QualifiedCapabilityExecutionIntentPreparationOutcome.CREATED


def test_identical_replay() -> None:
    preparation, acquisition_id, domain_ref, _, _ = _stack()
    req = _request(
        preparation=preparation,
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
    )
    preparation.prepare(req)
    result = preparation.prepare(req)
    assert (
        result.outcome
        is QualifiedCapabilityExecutionIntentPreparationOutcome.ALREADY_RECORDED_IDENTICAL
    )


def test_codecraft_not_applicable() -> None:
    preparation, acquisition_id, domain_ref, _, _ = _stack()
    req = _request(
        preparation=preparation,
        acquisition_id=acquisition_id,
        domain_ref=domain_ref,
    )
    qual = req.qualification_result.model_copy(
        update={"strategy_id": CODECRAFT_GAP_SYNTHESIS_STRATEGY_ID},
    )
    result = preparation.prepare(replace(req, qualification_result=qual))
    assert result.outcome is QualifiedCapabilityExecutionIntentPreparationOutcome.NOT_APPLICABLE


def test_wrong_tenant_integrity_failure() -> None:
    preparation, acquisition_id, domain_ref, _, _ = _stack(tenant_id="tenant-1")
    result = preparation.prepare(
        _request(
            preparation=preparation,
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
            tenant_id="tenant-other",
        ),
    )
    assert (
        result.outcome
        is QualifiedCapabilityExecutionIntentPreparationOutcome.INTEGRITY_FAILURE
    )


@dataclass
class _FailingIntentRepo:
    def record(self, intent):
        raise QualifiedMarketplaceToolExecutionIntentUnavailableError("down")

    def get(self, *, execution_request_id: str):
        return None


def test_repository_unavailable() -> None:
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    preparation, acquisition_id, domain_ref, _, _ = _stack()
    preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
        intent_repository=_FailingIntentRepo(),
        stage_repository=stage_repo,
        context_resolver=resolver,
    )
    result = preparation.prepare(
        _request(
            preparation=preparation,
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
    )
    assert result.outcome is QualifiedCapabilityExecutionIntentPreparationOutcome.UNAVAILABLE


@dataclass
class _ConflictIntentRepo:
    def record(self, intent):
        raise QualifiedMarketplaceToolExecutionIntentConflictError("conflict")

    def get(self, *, execution_request_id: str):
        return None


def test_repository_conflict() -> None:
    _, acquisition_id, domain_ref, stage_repo, resolver = _stack()
    preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
        intent_repository=_ConflictIntentRepo(),
        stage_repository=stage_repo,
        context_resolver=resolver,
    )
    result = preparation.prepare(
        _request(
            preparation=preparation,
            acquisition_id=acquisition_id,
            domain_ref=domain_ref,
        ),
    )
    assert result.outcome is QualifiedCapabilityExecutionIntentPreparationOutcome.CONFLICT
