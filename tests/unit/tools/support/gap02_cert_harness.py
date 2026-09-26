# © Artur Czarnecki. All rights reserved.

"""S24-GAP-02-CERT harness — composes production Marketplace→AW→EE stacks for certification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
)
from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CapabilityCatalogGovernedDiscoveryService,
    SkillRegistryManifestLookup,
)
from intergrax.autonomous_work.catalog_canonical_discovery_service import (
    CatalogCanonicalCapabilityDiscoveryService,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
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
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleKind,
    WorkerRecoveryDecision,
    derive_recovery_decision_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_catalog import CapabilityGovernanceContext
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.marketplace import MarketplaceQueryContext
from intergrax.contracts.marketplace.gap_acquisition import (
    MarketplaceGapAcquisitionRequest,
    MarketplaceGapAcquisitionResult,
)
from intergrax.contracts.tools.marketplace_handoff_reference import (
    derive_marketplace_gap_tool_handoff_id,
    marketplace_domain_handoff_reference,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
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
from intergrax.skills.registry.runtime import SkillRegistry
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
from intergrax.tools.marketplace_qualified_capability_qualification_provider import (
    MarketplaceToolCapabilityQualificationProvider,
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
from intergrax.tools.marketplace_qualified_tool_stage_context_recorder import (
    MarketplaceQualifiedToolStageContextRecorderImpl,
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
from intergrax.tools.qualified_tool_invocation_resolver import (
    DefaultQualifiedToolInvocationResolver,
)
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    Me14EchoInput,
)
from testing_support.marketplace_tool_execution_composition import (
    ME14_CAPABILITY_SOURCE,
    me14_default_listing_v1,
)
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.catalog_discovery_test_support import (
    catalog_discovery_dependencies,
    catalog_snapshot_from_registries,
    host_availability_for_entries,
    tool_catalog_entry,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import _READ
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    build_worker_execution_admission_for_uca6c,
)

_NOW = datetime(2026, 3, 26, 14, 0, tzinfo=UTC)
_PROFILE = CapabilityProfileRef(
    profile_id="cap/gap02-cert",
    version=initial_profile_version(),
)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_TASK_ID = TaskId("task_0000000000000000000000000000000c")
_HOST = "host-profile-gap02-cert"
_EVIDENCE = ProblemReference("problem/evidence/gap02-cert")


class _Material(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = "cert"


@dataclass
class _MaterialProvider:
    selected_operation: str = "invoke"

    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=Me14EchoInput(message=request.selected_operation),
        )


@dataclass
class RecordingInvoker:
    caller_agent_id: str = "agent-gap02-cert"
    calls: int = 0
    last_request: Any = None
    suspend: bool = False

    def invoke(self, request):
        self.calls += 1
        self.last_request = request
        if self.suspend:
            from intergrax.runtime.execution.suspended_operation.pause_required import (
                ExecutionSuspendedWorkPauseRequired,
            )

            raise ExecutionSuspendedWorkPauseRequired.__new__(
                ExecutionSuspendedWorkPauseRequired,
            )
        return ToolExecutionResult(success=True, output=None, error=None)


class CountingMaterializer(Me14ToolHostActivationMaterializer):
    def __init__(self, registry, *, catalog_source_id: str) -> None:
        super().__init__(registry, catalog_source_id=catalog_source_id)
        self.physical_activations = 0

    def materialize(self, resolution):
        self.physical_activations += 1
        return super().materialize(resolution)


class CountingGapAcquisitionPort:
    """Wraps gap service to count Marketplace gap acquisitions."""

    def __init__(self, inner: MarketplaceGapAcquisitionService) -> None:
        self._inner = inner
        self.acquire_calls = 0

    def acquire_from_gap(
        self,
        request: MarketplaceGapAcquisitionRequest,
    ) -> MarketplaceGapAcquisitionResult:
        self.acquire_calls += 1
        return self._inner.acquire_from_gap(request)


@dataclass
class CachedRecoveryPort:
    """Replays a prior canonical recovery outcome (restart / intent-resume proofs)."""

    outcome: WorkerCapabilityRecoveryOutcome
    calls: int = 0

    def coordinate_recovery(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        decided_at: datetime | None = None,
        allow_generic_acquisition: bool = True,
    ) -> WorkerCapabilityRecoveryOutcome:
        self.calls += 1
        return self.outcome


@dataclass
class QualificationCoordinatorAdapter:
    service: CapabilityQualificationService
    calls: int = 0

    def qualify(
        self,
        request: CapabilityQualificationRequest,
    ) -> CapabilityQualificationResult:
        self.calls += 1
        decision = self.service.qualify(request)
        return decision.qualification_result


@dataclass
class Gap02CertTrace:
    gap_id: str
    acquisition_request_id: str
    handoff_id: str
    domain_handoff_reference: str
    qualification_request_id: str
    recovery_decision_id: str
    discovery_correlation_id: str
    discovery_outcome: DiscoveryCompletionOutcome
    marketplace_acquire_calls_before_recovery: int
    marketplace_acquire_calls_after_recovery: int


@dataclass
class Gap02CertHarness:
    tenant_id: str
    store: InMemoryDocumentStore
    gap_port: CountingGapAcquisitionPort
    recovery: WorkerCapabilityRecoveryCoordinator
    qualification_adapter: QualificationCoordinatorAdapter
    stage_repo: DocumentStoreMarketplaceQualifiedToolStageRepository
    assoc_repo: DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository
    intent_repo: DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository
    binding_provider: MarketplaceToolQualifiedCapabilityBindingProvider
    activation_resolver: QualifiedMarketplaceToolActivationResolver
    materializer: CountingMaterializer
    lifecycle: ToolHostLifecycleService
    invoker: RecordingInvoker = field(default_factory=RecordingInvoker)

    @classmethod
    def build(
        cls,
        tenant_id: str,
        *,
        store: InMemoryDocumentStore | None = None,
        assoc_repo: DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository | None = None,
    ) -> Gap02CertHarness:
        resolved_store = store or InMemoryDocumentStore()
        listing = me14_default_listing_v1()
        source = MarketplaceCapabilityCatalogSource(
            source=ME14_CAPABILITY_SOURCE,
            records=(listing,),
        )
        catalog = FederatedCapabilityCatalog((source,))
        catalog_service = MarketplaceCatalogService(
            catalog=catalog,
            marketplace_sources=(source,),
        )
        stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(resolved_store)
        resolved_assoc = assoc_repo or DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
            resolved_store,
        )
        recorder = MarketplaceQualifiedToolStageContextRecorderImpl(resolved_assoc)
        consumer = ToolQualificationStagingConsumer(stage_repo, resolved_assoc)
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
        inner_gap = MarketplaceGapAcquisitionService(
            catalog_service=catalog_service,
            discovery_service=MarketplaceDiscoveryService.with_defaults(),
            governance_evaluators=(AvailabilityPreservingGovernanceEvaluator(),),
            governance_context=CapabilityGovernanceContext(),
            recommendation_service=MarketplaceRecommendationService.with_defaults(),
            handoff_orchestrator=orchestrator,
            tool_stage_context_recorder=recorder,
        )
        gap_port = CountingGapAcquisitionPort(inner_gap)
        strategy = MarketplaceGapCapabilityAcquisitionStrategy(
            gap_port,
            marketplace_query_context=MarketplaceQueryContext(tenant_id=tenant_id),
        )
        acquisition_service = CapabilityAcquisitionService(
            (strategy,),
            authorization=PermitCapabilityAcquisitionAuthorizationPort(),
        )
        tool_registry = ToolRegistry()
        skill_registry = SkillRegistry()
        snapshot = catalog_snapshot_from_registries(
            tool_registry=tool_registry,
            skill_registry=skill_registry,
        )
        availability = host_availability_for_entries(
            *(tool_catalog_entry(reg.contract.tool_id) for reg in tool_registry.list()),
        )
        dependencies = catalog_discovery_dependencies(
            snapshot=snapshot,
            availability_evidence=availability,
        )
        governed = CapabilityCatalogGovernedDiscoveryService(
            dependencies,
            manifest_lookup=SkillRegistryManifestLookup(skill_registry),
        )
        discovery = CatalogCanonicalCapabilityDiscoveryService(
            governed_discovery=governed,
        )
        resolver = MarketplaceQualifiedToolStageContextResolverImpl(resolved_assoc)
        qualification_service = CapabilityQualificationService(
            (
                MarketplaceToolCapabilityQualificationProvider(
                    stage_repository=stage_repo,
                    context_resolver=resolver,
                ),
            ),
        )
        qual_adapter = QualificationCoordinatorAdapter(service=qualification_service)
        recovery = WorkerCapabilityRecoveryCoordinator(
            discovery=discovery,
            acquisition=acquisition_service,
            authority_compatibility=AllowAllAuthorityCompatibilityPort(),
            qualification=qual_adapter,
        )
        binding_provider = MarketplaceToolQualifiedCapabilityBindingProvider(
            stage_repository=stage_repo,
            context_resolver=resolver,
        )
        intent_repo = DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository(
            resolved_store,
        )
        lifecycle = ToolHostLifecycleService(host_profile_id=_HOST)
        catalog_provider = Me14ToolCatalogProvider()
        materializer = CountingMaterializer(
            lifecycle.registry,
            catalog_source_id=catalog_provider.catalog_source_id,
        )
        acquisition_tools = DynamicToolAcquisitionService(
            catalog_registry=ToolCatalogProviderRegistry(
                {catalog_provider.catalog_source_id: catalog_provider},
            ),
            activation=lifecycle,
            materializer=materializer,
        )
        activation_resolver = QualifiedMarketplaceToolActivationResolver(
            activation_read=lifecycle,
            acquisition=acquisition_tools,
            host_profile_id=_HOST,
        )
        return cls(
            tenant_id=tenant_id,
            store=resolved_store,
            gap_port=gap_port,
            recovery=recovery,
            qualification_adapter=qual_adapter,
            stage_repo=stage_repo,
            assoc_repo=resolved_assoc,
            intent_repo=intent_repo,
            binding_provider=binding_provider,
            activation_resolver=activation_resolver,
            materializer=materializer,
            lifecycle=lifecycle,
        )

    def worker_need(self, recovery_decision_id: str) -> WorkerCapabilityNeed:
        obstacle_id = f"{_WORKER_ID}:obstacle:{recovery_decision_id}"
        return WorkerCapabilityNeed(
            worker_instance_id=_WORKER_ID,
            obstacle_id=obstacle_id,
            need_kind=CapabilityNeedKind.TOOL_OPERATION,
            required_operations=(ME14_TOOL_LOGICAL_ID,),
            capability_profile_ref=_PROFILE,
            requested_at=_NOW,
            recovery_decision_id=derive_recovery_decision_id(obstacle_id),
            evidence_refs=(_EVIDENCE,),
        )

    def recovery_decision(self, need: WorkerCapabilityNeed) -> WorkerRecoveryDecision:
        return WorkerRecoveryDecision(
            decision_id=derive_recovery_decision_id(need.obstacle_id),
            obstacle_id=need.obstacle_id,
            obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
            strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
            decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
            evidence_refs=(_EVIDENCE,),
            decided_at=_NOW,
            source_ref="recovery/gap02-cert",
        )

    def acquisition_request(
        self,
        need: WorkerCapabilityNeed,
        decision: WorkerRecoveryDecision,
    ) -> WorkerCapabilityAcquisitionRequest:
        return WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=decision,
            capability_profile_ref=_PROFILE,
        )

    def fulfillment_request(
        self,
        recovery_decision_id: str,
    ) -> WorkerCapabilityFulfillmentRequest:
        need = self.worker_need(recovery_decision_id)
        decision = self.recovery_decision(need)
        return WorkerCapabilityFulfillmentRequest(
            acquisition_request=self.acquisition_request(need, decision),
            worker_instance_id=_WORKER_ID,
            tenant_id=self.tenant_id,
            task_id=_TASK_ID,
            requested_at=_NOW,
            requested_authority_scopes=(_READ,),
            allow_generic_acquisition=True,
        )

    def run_true_gap_recovery(
        self,
        recovery_decision_id: str,
    ) -> tuple[WorkerCapabilityRecoveryOutcome, Gap02CertTrace]:
        before = self.gap_port.acquire_calls
        need = self.worker_need(recovery_decision_id)
        request = self.acquisition_request(need, self.recovery_decision(need))
        outcome = self.recovery.coordinate_recovery(request, decided_at=_NOW)
        after = self.gap_port.acquire_calls
        assert before == 0, "Marketplace acquisition before TRUE GAP must be 0"
        completion = outcome.discovery_completion
        assert completion is not None
        assert completion.outcome is DiscoveryCompletionOutcome.MISSING_CAPABILITY
        assert outcome.acquisition_result is not None
        acq = outcome.acquisition_result
        assert acq.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
        assert acq.strategy_id == MARKETPLACE_GAP_ACQUISITION_STRATEGY_ID
        assert acq.evidence is not None
        assert acq.evidence.domain_handoff_reference is not None
        assert acq.evidence.evidence_ref is not None
        assert acq.evidence.artifact_reference is None
        handoff_id = derive_marketplace_gap_tool_handoff_id(
            tenant_id=self.tenant_id,
            operation_id=acq.request_id,
        )
        domain_ref = marketplace_domain_handoff_reference(handoff_id)
        assert acq.evidence.domain_handoff_reference == domain_ref
        assoc = self.assoc_repo.get_by_handoff_id(handoff_id=handoff_id)
        assert assoc is not None
        assert assoc.tenant_id == self.tenant_id
        assert assoc.acquisition_request_id == acq.request_id
        assert assoc.handoff_id == handoff_id
        stage = self.stage_repo.get(tenant_id=self.tenant_id, handoff_id=handoff_id)
        assert stage is not None
        assert stage.tenant_id == self.tenant_id
        assert stage.handoff_id == handoff_id
        assert stage.selected_release.discovery.logical.logical_id == ME14_TOOL_LOGICAL_ID
        assert stage.selected_release.package_reference == ME14_PACKAGE_REFERENCE_V1
        assert stage.selected_release.version_label == ME14_VERSION_V1
        assert stage.selected_release.content_digest == ME14_DIGEST_V1
        assert self.materializer.physical_activations == 0
        assert len(self.lifecycle.registry.list()) == 0
        assert outcome.phase is WorkerCapabilityRecoveryPhase.QUALIFICATION_COMPLETE
        assert outcome.qualification_result is not None
        assert (
            outcome.qualification_result.outcome is CapabilityQualificationOutcome.QUALIFIED
        )
        qual_id = outcome.provenance.qualification_request_id
        assert qual_id is not None
        trace = Gap02CertTrace(
            gap_id=acq.gap_id,
            acquisition_request_id=acq.request_id,
            handoff_id=handoff_id,
            domain_handoff_reference=domain_ref,
            qualification_request_id=qual_id,
            recovery_decision_id=recovery_decision_id,
            discovery_correlation_id=completion.discovery_correlation_id,
            discovery_outcome=completion.outcome,
            marketplace_acquire_calls_before_recovery=before,
            marketplace_acquire_calls_after_recovery=after,
        )
        return outcome, trace

    def build_fulfillment_coordinator(
        self,
        invoker: RecordingInvoker | None = None,
        *,
        recovery: WorkerCapabilityRecoveryCoordinator | CachedRecoveryPort | None = None,
    ) -> WorkerCapabilityFulfillmentCoordinator:
        resolved_invoker = invoker or self.invoker
        self.invoker = resolved_invoker
        context_resolver = MarketplaceQualifiedToolStageContextResolverImpl(self.assoc_repo)
        intent_preparation = MarketplaceQualifiedToolExecutionIntentPreparation(
            intent_repository=self.intent_repo,
            stage_repository=self.stage_repo,
            context_resolver=context_resolver,
        )
        handler = MarketplaceToolQualifiedCapabilityExecutionHandler(
            intent_repository=self.intent_repo,
            stage_repository=self.stage_repo,
            activation_resolver=self.activation_resolver,
            material_provider=_MaterialProvider(),
            invocation_resolver=DefaultQualifiedToolInvocationResolver(),
            catalog_tool_invoker=resolved_invoker,
        )
        dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
            handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
            runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        )
        execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
        resume = WorkerQualifiedCapabilityResumeCoordinator(
            binding=QualifiedCapabilityBindingService((self.binding_provider,)),
            execution=execution,
            authority_admission=build_worker_execution_admission_for_uca6c(
                worker_instance_id=_WORKER_ID,
                tenant_id=self.tenant_id,
                workspace_id="workspace-gap02-cert",
                principal_id="principal-gap02-cert",
            ),
        )

        @dataclass
        class _DirectReuse:
            def fulfill(self, request):
                raise AssertionError("direct reuse not used in cert")

        resolved_recovery = recovery if recovery is not None else self.recovery
        return WorkerCapabilityFulfillmentCoordinator(
            recovery=resolved_recovery,
            resume=resume,
            direct_reuse=_DirectReuse(),
            intent_preparation=intent_preparation,
        )

    def reconstruct_tool_domain(self) -> Gap02CertHarness:
        """New service instances over the same durable document store (restart simulation)."""
        return Gap02CertHarness.build(
            self.tenant_id,
            store=self.store,
            assoc_repo=DocumentStoreMarketplaceQualifiedToolStageContextAssociationRepository(
                self.store,
            ),
        )


__all__ = [
    "CountingGapAcquisitionPort",
    "CountingMaterializer",
    "Gap02CertHarness",
    "Gap02CertTrace",
    "RecordingInvoker",
    "_NOW",
    "_TASK_ID",
    "_WORKER_ID",
]
