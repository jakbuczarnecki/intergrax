# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-P3 canonical AW fulfillment → EE → ToolRuntime integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
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
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
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
)
from intergrax.contracts.tools.marketplace_qualified_tool_stage_context import (
    MarketplaceQualifiedToolStageContext,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionService
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MarketplaceToolQualifiedCapabilityBindingProvider,
)
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
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
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    Me14EchoInput,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
    _recovery_decision,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _READ,
    _TENANT,
    _WORKER_ID,
    _authority_admission,
)
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import (
    _qualification,
    _subject,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 3, 26, 12, 0, tzinfo=UTC)
_TASK_ID = TaskId("task_00000000000000000000000000000002")
_HOST = "host-profile-me14-p3"
_ACQUISITION_ID = "acq-gap02-p3"
_QUAL_REQUEST = "qual-req-gap02-p3"
_GAP = "gap-1"
_STRATEGY = "marketplace.gap_acquisition.v1"
_ME14_SOURCE = CapabilitySourceIdentity(
    source_id="official.intergrax.me14",
    source_kind=CapabilitySourceKind.OFFICIAL,
)


class _Material(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = "hello"


@dataclass
class _MaterialProvider:
    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=Me14EchoInput(message=request.selected_operation),
        )


@dataclass
class _RecordingInvoker:
    caller_agent_id: str = "agent-gap02"
    calls: int = 0

    def invoke(self, request):
        self.calls += 1
        return ToolExecutionResult(success=True, output=None, error=None)


class _CountingMaterializer(Me14ToolHostActivationMaterializer):
    def __init__(self, registry, *, catalog_source_id: str) -> None:
        super().__init__(registry, catalog_source_id=catalog_source_id)
        self.physical_activations = 0

    def materialize(self, resolution):
        self.physical_activations += 1
        return super().materialize(resolution)


def _me14_release() -> CapabilityReleaseIdentity:
    return CapabilityReleaseIdentity(
        discovery=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=_ME14_SOURCE,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=ME14_TOOL_LOGICAL_ID,
            ),
        ),
        publisher="publisher:me14",
        version_label=ME14_VERSION_V1,
        content_digest=ME14_DIGEST_V1,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
    )


def _marketplace_stack() -> tuple[
    InMemoryDocumentStore,
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
    MarketplaceToolQualifiedCapabilityBindingProvider,
    QualifiedMarketplaceToolActivationResolver,
    _CountingMaterializer,
    str,
    str,
]:
    store = InMemoryDocumentStore()
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    resolver_ctx = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    binding_provider = MarketplaceToolQualifiedCapabilityBindingProvider(
        stage_repository=stage_repo,
        context_resolver=resolver_ctx,
    )
    handoff_id = derive_marketplace_gap_tool_handoff_id(
        tenant_id=_TENANT,
        operation_id=_ACQUISITION_ID,
    )
    domain_ref = marketplace_domain_handoff_reference(handoff_id)
    assoc_repo.record(
        MarketplaceQualifiedToolStageContext(
            handoff_id=handoff_id,
            tenant_id=_TENANT,
            acquisition_request_id=_ACQUISITION_ID,
        ),
    )
    stage_repo.stage(
        MarketplaceQualifiedToolStage(
            handoff_id=handoff_id,
            tenant_id=_TENANT,
            selected_release=_me14_release(),
            discovery_correlation_id="discovery-gap02",
            selection_id=f"marketplace-gap-selection:{_ACQUISITION_ID}",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=_NOW,
        ),
    )
    intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(store)
    lifecycle = ToolHostLifecycleService(host_profile_id=_HOST)
    catalog_provider = Me14ToolCatalogProvider()
    materializer = _CountingMaterializer(
        lifecycle.registry,
        catalog_source_id=catalog_provider.catalog_source_id,
    )
    acquisition = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry(
            {catalog_provider.catalog_source_id: catalog_provider},
        ),
        activation=lifecycle,
        materializer=materializer,
    )
    activation_resolver = QualifiedMarketplaceToolActivationResolver(
        activation_read=lifecycle,
        acquisition=acquisition,
        host_profile_id=_HOST,
    )
    return (
        store,
        intent_repo,
        binding_provider,
        activation_resolver,
        materializer,
        domain_ref,
        handoff_id,
    )


def _marketplace_acquisition(domain_ref: str) -> CapabilityAcquisitionResult:
    return CapabilityAcquisitionResult(
        request_id=_ACQUISITION_ID,
        gap_id=_GAP,
        strategy_id=_STRATEGY,
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_NOW,
        completed_at=_NOW,
        evidence=CapabilityAcquisitionEvidence(domain_handoff_reference=domain_ref),
    )


def _marketplace_provenance() -> WorkerCapabilityRecoveryProvenance:
    return WorkerCapabilityRecoveryProvenance(
        worker_need_id="worker-need:gap02-p3",
        canonical_need_id="capability-need:gap02-p3",
        discovery_correlation_id="aw-canonical-discovery:gap02-p3",
        discovery_completion_outcome="missing_capability",
        gap_id=_GAP,
        acquisition_request_id=_ACQUISITION_ID,
        acquisition_strategy_id=_STRATEGY,
        qualification_request_id=_QUAL_REQUEST,
    )


@dataclass
class _StaticRecovery:
    outcome: WorkerCapabilityRecoveryOutcome

    def coordinate_recovery(self, request, *, decided_at, allow_generic_acquisition):
        return self.outcome


def _need(recovery_decision_id: str) -> WorkerCapabilityNeed:
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:{recovery_decision_id}",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("invoke",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=recovery_decision_id,
    )


def _fulfillment_request(recovery_decision_id: str) -> WorkerCapabilityFulfillmentRequest:
    need = _need(recovery_decision_id)
    decision = _recovery_decision(need)
    return WorkerCapabilityFulfillmentRequest(
        acquisition_request=WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=decision,
            capability_profile_ref=_PROFILE,
        ),
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        requested_at=_NOW,
        requested_authority_scopes=(_READ,),
        allow_generic_acquisition=False,
    )


@dataclass
class _DirectReusePort:
    def fulfill(self, request):
        raise AssertionError("not used")


def _build_coordinator(
    *,
    recovery_decision_id: str,
    invoker: _RecordingInvoker,
    intent_repo: DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
    binding_provider: MarketplaceToolQualifiedCapabilityBindingProvider,
    activation_resolver: QualifiedMarketplaceToolActivationResolver,
    domain_ref: str,
    store: InMemoryDocumentStore,
) -> WorkerCapabilityFulfillmentCoordinator:
    qualification = _qualification(
        acquisition_id=_ACQUISITION_ID,
        domain_ref=domain_ref,
        qualification_request_id=_QUAL_REQUEST,
    )
    acquisition = _marketplace_acquisition(domain_ref)
    recovery = _StaticRecovery(
        outcome=WorkerCapabilityRecoveryOutcome(
            phase=WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE,
            provenance=_marketplace_provenance(),
            acquisition_result=acquisition,
            qualification_result=qualification,
            decided_at=_NOW,
        ),
    )
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    assoc_repo = DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
        store,
    )
    context_resolver = MarketplaceQualifiedToolStageContextResolverImpl(assoc_repo)
    intent_preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
        intent_repository=intent_repo,
        stage_repository=stage_repo,
        context_resolver=context_resolver,
    )
    handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=intent_repo,
        stage_repository=stage_repo,
        activation_resolver=activation_resolver,
        material_provider=_MaterialProvider(),
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        catalog_tool_invoker=invoker,
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    resume = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService((binding_provider,)),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    return WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery,
        resume=resume,
        direct_reuse=_DirectReusePort(),
        intent_preparation=intent_preparation,
    )


def _execution_request_id_for(recovery_decision_id: str, domain_ref: str) -> str:
    qualification = _qualification(
        acquisition_id=_ACQUISITION_ID,
        domain_ref=domain_ref,
        qualification_request_id=_QUAL_REQUEST,
    )
    subject = _subject(domain_ref, qualification_request_id=_QUAL_REQUEST)
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=recovery_decision_id,
        qualification_request_id=qualification.qualification_request_id,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    return derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )


def test_canonical_fulfillment_path_first_execution_dispatched() -> None:
    store, intent_repo, binding_provider, activation_resolver, materializer, domain_ref, _ = (
        _marketplace_stack()
    )
    invoker = _RecordingInvoker()
    recovery_decision_id = "recovery:gap02:p3:a"
    coordinator = _build_coordinator(
        recovery_decision_id=recovery_decision_id,
        invoker=invoker,
        intent_repo=intent_repo,
        binding_provider=binding_provider,
        activation_resolver=activation_resolver,
        domain_ref=domain_ref,
        store=store,
    )
    execution_request_id = _execution_request_id_for(recovery_decision_id, domain_ref)
    result = coordinator.fulfill(_fulfillment_request(recovery_decision_id))
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert intent_repo.get(execution_request_id=execution_request_id) is not None
    assert materializer.physical_activations == 1
    assert invoker.calls == 1


def test_canonical_two_executions_one_physical_activation() -> None:
    store, intent_repo, binding_provider, activation_resolver, materializer, domain_ref, _ = (
        _marketplace_stack()
    )
    invoker = _RecordingInvoker()
    recovery_a = "recovery:gap02:p3:exec-a"
    recovery_b = "recovery:gap02:p3:exec-b"
    coordinator_a = _build_coordinator(
        recovery_decision_id=recovery_a,
        invoker=invoker,
        intent_repo=intent_repo,
        binding_provider=binding_provider,
        activation_resolver=activation_resolver,
        domain_ref=domain_ref,
        store=store,
    )
    coordinator_b = _build_coordinator(
        recovery_decision_id=recovery_b,
        invoker=invoker,
        intent_repo=intent_repo,
        binding_provider=binding_provider,
        activation_resolver=activation_resolver,
        domain_ref=domain_ref,
        store=store,
    )
    exec_a = _execution_request_id_for(recovery_a, domain_ref)
    exec_b = _execution_request_id_for(recovery_b, domain_ref)
    assert exec_a != exec_b

    result_a = coordinator_a.fulfill(_fulfillment_request(recovery_a))
    result_b = coordinator_b.fulfill(_fulfillment_request(recovery_b))

    assert result_a.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert result_b.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert intent_repo.get(execution_request_id=exec_a) is not None
    assert intent_repo.get(execution_request_id=exec_b) is not None
    assert materializer.physical_activations == 1
    assert invoker.calls == 2
