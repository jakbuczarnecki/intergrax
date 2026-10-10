# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R1 production Marketplace host composition proofs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.applications._shared.uca6c_marketplace_qualified_execution_composition import (
    Uca6cMarketplaceQualifiedExecutionCompositionError,
    build_production_marketplace_configured_execution_bound_dispatch,
    build_production_marketplace_configured_execution_composition,
    build_production_marketplace_configured_execution_fulfillment,
    build_production_marketplace_qualified_capability_execution_dispatch,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
    WorkerConfiguredCapabilityFulfillmentService,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityProfileRef,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.autonomous_work.ids import mint_worker_instance_id
from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityResumeRequest,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_execution_id,
    reset_active_execution_identity,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenanceMode,
)
from intergrax.contracts.tools.marketplace_tool_execution_intent import (
    MarketplaceToolExecutionIntent,
    UcaMarketplaceToolExecutionProvenance,
)
from intergrax.contracts.tools.qualified_tool_invocation import (
    QualifiedToolInvocationMaterialOutcome,
    QualifiedToolInvocationMaterialRequest,
    QualifiedToolInvocationMaterialResult,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationResult,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationMaterializationPort,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.sandbox.isolation_gate import SandboxIsolationAvailability
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
)
from intergrax.tools.marketplace_qualified_capability_binding_provider import (
    MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
    MarketplaceToolQualifiedCapabilityBindingProvider,
    execution_target_reference_for_marketplace_qualified_tool,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    build_marketplace_tool_execution_target,
)
from intergrax.tools.marketplace_qualified_capability_staging import (
    DocumentStoreMarketplaceQualifiedToolStageRepository,
)
from intergrax.tools.providers.database.contracts import DatabaseQueryInput
from intergrax.tools.providers.database.service import DATABASE_QUERY_TOOL_ID
from intergrax.tools.providers.database.bundle import register_database_tools
from intergrax.tools.qualified_marketplace_tool_execution_intent_repository import (
    DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository,
)
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.autonomous_work.worker_recovery_governed_fulfillment_composition import (
    build_worker_recovery_governed_fulfillment_wiring,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.in_memory_worker_recovery_obstacle_capability_need_repository import (
    InMemoryWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.autonomous_work.capability_acquisition_ports import (
    StaticWorkerCapabilityProfileResolver,
    permissive_capability_policy,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    InMemoryKVStore,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _READ,
    _PRINCIPAL,
    _WORKSPACE,
)
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    build_worker_execution_admission_for_uca6c,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _TENANT,
    _acquisition,
    _provenance,
)
from tests.unit.autonomous_work.test_uca6c_worker_qualified_capability_resume import (
    _qualification,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResult,
)
from tests.unit.autonomous_work.test_uca6b_worker_capability_recovery import (
    _PROFILE,
)
from tests.unit.tools.test_marketplace_gap02_p3_integration import (
    _HOST,
    _marketplace_acquisition,
    _marketplace_stack,
    _subject,
)
from tests.unit.tools.test_marketplace_qualified_capability_binding_provider import _release
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_00000000000000000000000000000003")
_RUN_ID = RunId("run_" + "d" * 32)
_ATTEMPT_ID = AttemptId("attempt_" + "e" * 32)
_NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
_CONFIG_REF = ConfigurationOpportunityRef("cfg/opportunity-p3-r1")


class _NonConditionalDocumentStore(DocumentStore):
    def put(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def get(self, *args: object, **kwargs: object) -> None:
        return None

    def delete(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


class _FakeRelationalClient:
    def __init__(self, token: object) -> None:
        self.token = token
        self.fetch_calls = 0

    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        return None

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> Sequence[Mapping[str, Any]]:
        self.fetch_calls += 1
        return ({"v": 1},)

    def close(self) -> None:
        return None


class _FakeRelationalIntegration(RelationalStoreIntegrationContract):
    def __init__(self, token: object, *, provider_id: str = "sqlite") -> None:
        super().__init__(
            **RelationalStoreIntegrationContract.for_provider(
                provider_id=provider_id,
                display_name="Fake Relational",
            ).model_dump(),
        )
        self._client = _FakeRelationalClient(token)

    def connect(self) -> None:
        self._client.connect()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._client.execute(sql, params)

    def fetch_all(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> Sequence[Mapping[str, Any]]:
        return self._client.fetch_all(sql, params)

    @property
    def io_calls(self) -> int:
        return self._client.fetch_calls

    def close(self) -> None:
        self._client.close()


class _CountingMaterialization(ExecutionBoundIntegrationMaterializationPort):
    def __init__(self, instance: RelationalStoreIntegrationContract) -> None:
        self.count = 0
        self.instance = instance
        self.last_materialized: RelationalStoreIntegrationContract | None = None

    def resolve_catalog(
        self,
        category: IntegrationCategory,
        *,
        slug: str,
        profile=None,
    ) -> RelationalStoreIntegrationContract:
        self.count += 1
        self.last_materialized = self.instance
        return self.instance

    def resolve_from_profile(self, profile, category: IntegrationCategory):
        self.count += 1
        self.last_materialized = self.instance
        return self.instance


class _DatabaseMaterialProvider:
    def provide(self, request: QualifiedToolInvocationMaterialRequest):
        return QualifiedToolInvocationMaterialResult(
            outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
            material=DatabaseQueryInput(sql="SELECT 1"),
        )


@dataclass
class _DatabaseActivationResolver:
    def ensure_exact_active(self, *, stage: object, execution_request_id: str):
        return QualifiedMarketplaceToolActivationResult(
            outcome=QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
            registry_tool_id=DATABASE_QUERY_TOOL_ID,
        )

    def ensure_exact_active_for_identity(
        self,
        *,
        capability_identity: CapabilityIdentityKey,
        execution_request_id: str,
        package_resolver: object,
    ):
        _ = capability_identity, execution_request_id, package_resolver
        return QualifiedMarketplaceToolActivationResult(
            outcome=QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
            registry_tool_id=DATABASE_QUERY_TOOL_ID,
        )


@dataclass
class _MarketplaceDeps:
    store: InMemoryDocumentStore
    intent_repo: DocumentStoreQualifiedMarketplaceToolExecutionIntentRepository
    stage_repo: DocumentStoreMarketplaceQualifiedToolStageRepository
    binding_provider: MarketplaceToolQualifiedCapabilityBindingProvider
    domain_ref: str
    handoff_id: str


def _sandbox_availability() -> SandboxIsolationAvailability:
    return SandboxIsolationAvailability(
        session_configured=True,
        host_configured=True,
        healthy=True,
    )


def _governed_database_catalog_invoker(
    *,
    allowed_tools: set[str] | None = None,
    allowed_capabilities: frozenset[str] | None = None,
    grant_agent_id: str = "marketplace-p3-r1-agent",
) -> object:
    registry = ToolRegistry()
    ctx = ToolWiringContext()
    register_database_tools(registry, ctx)
    allowed = allowed_tools or {DATABASE_QUERY_TOOL_ID}
    composition = build_execution_bound_catalog_tool_composition(
        registry=registry,
        policy_bundle=RuntimePolicyBundle(),
        caller_agent_id="marketplace-p3-r1-agent",
        sandbox_availability=_sandbox_availability,
        production_mode=False,
        scope_policy=StaticToolScopePolicy(allowed_tools=allowed),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=(
                CapabilityGrant(
                    agent_id=grant_agent_id,
                    tenant_id=_TENANT,
                    allowed_capabilities=allowed_capabilities
                    or frozenset({"database"}),
                ),
            ),
        ),
        canonical_inner_execution_guard=None,
        meaningful_side_effect_authorization=None,
        document_store=None,
        continuation_dependencies=None,
        reentry_claim_owner_id="marketplace-p3-r1",
    )
    return composition.invoker


def _marketplace_database_deps() -> _MarketplaceDeps:
    store, intent_repo, binding_provider, _activation, _materializer, domain_ref, handoff_id = (
        _marketplace_stack()
    )
    stage_repo = DocumentStoreMarketplaceQualifiedToolStageRepository(store)
    return _MarketplaceDeps(
        store=store,
        intent_repo=intent_repo,
        stage_repo=stage_repo,
        binding_provider=binding_provider,
        domain_ref=domain_ref,
        handoff_id=handoff_id,
    )


def _production_runtime_event_bus() -> RuntimeEventBus:
    return RuntimeEventBus(persistence=InMemoryRuntimeEventStore())


def _production_composition(
    *,
    materialization: _CountingMaterialization,
    catalog_invoker: object,
    deps: _MarketplaceDeps,
    runtime_event_bus: RuntimeEventBus | None = None,
):
    return build_production_marketplace_configured_execution_composition(
        intent_repository=deps.intent_repo,
        stage_repository=deps.stage_repo,
        activation_read=MagicMock(),
        acquisition=MagicMock(),
        host_profile_id=_HOST,
        material_provider=_DatabaseMaterialProvider(),
        catalog_tool_invoker=catalog_invoker,
        configuration_pinning_kv_store=InMemoryKVStore(),
        materialization=materialization,
        invocation_resolver=DefaultQualifiedToolInvocationResolver(),
        activation_resolver=_DatabaseActivationResolver(),
        package_resolver=MagicMock(),
        runtime_event_bus=runtime_event_bus or _production_runtime_event_bus(),
    )


def _with_active_execution_identity(execution_id: ExecutionId, fn: object):
    identity_token = bind_active_execution_identity(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=execution_id,
        task_id=_TASK_ID,
    )
    governance_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-1",
            principal_id="principal-1",
        ),
    )
    try:
        return fn()
    finally:
        reset_active_execution_governance_identity(governance_token)
        reset_active_execution_identity(identity_token)


def _adoption(tenant: str = _TENANT) -> ExecutionIntegrationConfigurationAdoption:
    binding = ConfiguredCapabilityBinding(
        tenant_id=tenant,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        configuration_type="test",
        configuration_version="v1",
        configuration_fingerprint="fp",
        realization_evidence_refs=(),
    )
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=binding,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        resource_scope="default",
    )


def _record_marketplace_intent(
    deps: _MarketplaceDeps,
    *,
    tenant: str = _TENANT,
    task_id: TaskId = _TASK_ID,
) -> MarketplaceToolExecutionIntent:
    subject_ref = (
        "qualified-capability-subject:q:"
        f"{deps.domain_ref}:h:{deps.handoff_id}"
    )
    resume_id = "resume-p3-r1"
    binding_id = "bind-p3-r1"
    execution_request_id = "exec-req-p3-r1"
    from intergrax.contracts.marketplace.handoff_traceability import (
        CapabilityHandoffConsumerTarget,
    )
    from intergrax.contracts.tools.marketplace_qualified_capability import (
        MarketplaceQualifiedToolStage,
    )

    release = _release()
    deps.stage_repo.stage(
        MarketplaceQualifiedToolStage(
            handoff_id=deps.handoff_id,
            tenant_id=tenant,
            selected_release=release,
            discovery_correlation_id="discovery-p3-r1",
            selection_id="selection-p3-r1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=_NOW,
        ),
    )
    capability_identity = CapabilityIdentityKey.from_discovery_identity(
        release.discovery,
    )
    intent = MarketplaceToolExecutionIntent(
        execution_request_id=execution_request_id,
        binding_operation_id=binding_id,
        tenant_id=tenant,
        task_id=str(task_id),
        worker_need_id="worker-need-p3-r1",
        subject_reference=subject_ref,
        capability_identity=capability_identity,
        selected_operation="query",
        provenance=UcaMarketplaceToolExecutionProvenance(
            handoff_id=deps.handoff_id,
            resume_operation_id=resume_id,
            uca_qualified_subject_reference=subject_ref,
        ),
    )
    deps.intent_repo.record(intent)
    return intent


def _minimal_marketplace_handler_kwargs(deps: _MarketplaceDeps) -> dict[str, object]:
    return {
        "intent_repository": deps.intent_repo,
        "stage_repository": deps.stage_repo,
        "activation_read": MagicMock(),
        "acquisition": MagicMock(),
        "host_profile_id": _HOST,
        "material_provider": _DatabaseMaterialProvider(),
        "catalog_tool_invoker": _governed_database_catalog_invoker(),
    }


def test_production_builder_requires_durable_backing() -> None:
    with pytest.raises(Uca6cMarketplaceQualifiedExecutionCompositionError):
        build_production_marketplace_configured_execution_composition(
            **_minimal_marketplace_handler_kwargs(_marketplace_database_deps()),
        )


def test_production_builder_rejects_two_backings() -> None:
    deps = _marketplace_database_deps()
    with pytest.raises(Uca6cMarketplaceQualifiedExecutionCompositionError):
        build_production_marketplace_configured_execution_composition(
            **_minimal_marketplace_handler_kwargs(deps),
            configuration_pinning_kv_store=InMemoryKVStore(),
            configuration_pinning_document_store=InMemoryDocumentStore(),
        )


def test_production_builder_rejects_non_conditional_document_store() -> None:
    deps = _marketplace_database_deps()
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        build_production_marketplace_configured_execution_composition(
            **_minimal_marketplace_handler_kwargs(deps),
            configuration_pinning_document_store=_NonConditionalDocumentStore(),
            runtime_event_bus=_production_runtime_event_bus(),
        )


def test_configured_adoption_durable_pin_and_same_provider_instance() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    deps = _marketplace_database_deps()
    invoker = _governed_database_catalog_invoker()
    composition = _production_composition(
        materialization=materialization,
        catalog_invoker=invoker,
        deps=deps,
    )

    intent = _record_marketplace_intent(deps)
    execution_id = mint_execution_id()
    target = intent.qualified_subject_reference
    dispatch_request = BoundCapabilityExecutionDispatchRequest(
        execution_request_id=intent.execution_request_id,
        execution_target=build_marketplace_tool_execution_target(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                deps.handoff_id,
            ),
            binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
            qualified_subject_reference=target,
        ),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
    )
    result = _with_active_execution_identity(
        execution_id,
        lambda: composition.handler.dispatch_once(
            dispatch_request,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=execution_id,
            integration_configuration_adoption=_adoption(),
        ),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    assert materialization.count == 1
    assert materialization.last_materialized is integration
    assert integration.io_calls == 1
    pins = composition.pinning_store.read_all(
        tenant_id=_TENANT,
        execution_id=execution_id,
    )
    assert len(pins) == 1
    pin = pins[0]
    assert pin.mode is ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED
    assert pin.configured is not None
    assert pin.configured.provider_id == "sqlite"
    assert pin.effective.provider_id == "sqlite"


def test_governance_scope_deny_zero_materialization_and_pin() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    deps = _marketplace_database_deps()
    invoker = _governed_database_catalog_invoker(grant_agent_id="denied-agent")
    composition = _production_composition(
        materialization=materialization,
        catalog_invoker=invoker,
        deps=deps,
    )
    intent = _record_marketplace_intent(deps)
    execution_id = mint_execution_id()
    from intergrax.contracts.capability_qualification.qualified_capability_binding import (
        QualifiedCapabilityExecutionTarget,
    )

    dispatch_request = BoundCapabilityExecutionDispatchRequest(
        execution_request_id=intent.execution_request_id,
        execution_target=build_marketplace_tool_execution_target(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                deps.handoff_id,
            ),
            binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
            qualified_subject_reference=intent.qualified_subject_reference,
        ),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
    )
    from intergrax.runtime.agent_governance.errors import CapabilityNotGrantedError

    with pytest.raises(CapabilityNotGrantedError):
        _with_active_execution_identity(
            execution_id,
            lambda: composition.handler.dispatch_once(
                dispatch_request,
                run_id=_RUN_ID,
                attempt_id=_ATTEMPT_ID,
                execution_id=execution_id,
                integration_configuration_adoption=_adoption(),
            ),
        )
    assert materialization.count == 0
    assert integration.io_calls == 0
    assert (
        composition.pinning_store.read_all(tenant_id=_TENANT, execution_id=execution_id) == ()
    )


def test_tenant_mismatch_blocks_materialization_pin_and_io() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    deps = _marketplace_database_deps()
    composition = _production_composition(
        materialization=materialization,
        catalog_invoker=_governed_database_catalog_invoker(),
        deps=deps,
    )
    intent = _record_marketplace_intent(deps, tenant="tenant-a")
    execution_id = mint_execution_id()
    from intergrax.contracts.capability_qualification.qualified_capability_binding import (
        QualifiedCapabilityExecutionTarget,
    )

    dispatch_request = BoundCapabilityExecutionDispatchRequest(
        execution_request_id=intent.execution_request_id,
        execution_target=build_marketplace_tool_execution_target(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                deps.handoff_id,
            ),
            binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
            qualified_subject_reference=intent.qualified_subject_reference,
        ),
        tenant_id="tenant-b",
        task_id=_TASK_ID,
    )
    result = _with_active_execution_identity(
        execution_id,
        lambda: composition.handler.dispatch_once(
            dispatch_request,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=execution_id,
            integration_configuration_adoption=_adoption(tenant="tenant-a"),
        ),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert materialization.count == 0
    assert integration.io_calls == 0
    assert composition.pinning_store.read_all(tenant_id="tenant-a", execution_id=execution_id) == ()
    assert composition.pinning_store.read_all(tenant_id="tenant-b", execution_id=execution_id) == ()


@dataclass
class _RecordingCatalogInvoker:
    caller_agent_id: str = "marketplace-p3-r1-agent"
    calls: int = 0

    def invoke(self, request: object) -> object:
        from intergrax.tools.execution_models import ToolExecutionResult

        self.calls += 1
        return ToolExecutionResult(success=True, output=None, error=None)


def test_ordinary_non_configured_marketplace_execution_still_works() -> None:
    deps = _marketplace_database_deps()
    invoker = _RecordingCatalogInvoker()
    composition = _production_composition(
        materialization=_CountingMaterialization(_FakeRelationalIntegration(object())),
        catalog_invoker=invoker,
        deps=deps,
    )
    intent = _record_marketplace_intent(deps)
    from intergrax.contracts.capability_qualification.qualified_capability_binding import (
        QualifiedCapabilityExecutionTarget,
    )

    dispatch_request = BoundCapabilityExecutionDispatchRequest(
        execution_request_id=intent.execution_request_id,
        execution_target=build_marketplace_tool_execution_target(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                deps.handoff_id,
            ),
            binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
            qualified_subject_reference=intent.qualified_subject_reference,
        ),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
    )
    execution_id = mint_execution_id()
    result = _with_active_execution_identity(
        execution_id,
        lambda: composition.handler.dispatch_once(
            dispatch_request,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=execution_id,
            integration_configuration_adoption=None,
        ),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    assert invoker.calls == 1


def test_configure_existing_adoption_reaches_production_handler() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    deps = _marketplace_database_deps()
    composition = _production_composition(
        materialization=materialization,
        catalog_invoker=_governed_database_catalog_invoker(),
        deps=deps,
    )
    intent = _record_marketplace_intent(deps)
    binding = ConfiguredCapabilityBinding(
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        configuration_type="test",
        configuration_version="v1",
        configuration_fingerprint="fp",
        realization_evidence_refs=("evidence-1",),
    )
    opportunity = ExistingCapabilityConfigurationOpportunity(
        configuration_ref=_CONFIG_REF,
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        current_revision="rev-1",
        configuration=MagicMock(
            configuration_type="test",
            configuration_version="v1",
            configuration_fingerprint="fp",
        ),
        configuration_fingerprint="fp",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    read = MagicMock()
    read.read_exact.return_value = opportunity
    realize = MagicMock()
    realize.realize.return_value = ExistingCapabilityConfigurationRealizationResult(
        request_id="req-p3-r1",
        configured_binding=binding,
        authorization_evidence=MagicMock(),
    )
    principal = MagicMock()
    principal.tenant_id = _TENANT
    principal.principal_id = "principal-1"
    resolver = MagicMock()
    resolver.resolve.return_value = principal
    configured = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=realize,
        principal_binding_resolver=resolver,
    )
    adoption = configured.fulfill_configure_existing(
        MagicMock(
            tenant_id=_TENANT,
            run_id=None,
            task_id=_TASK_ID,
            acquisition_request=MagicMock(
                need=MagicMock(recovery_decision_id="recovery-p3-r1"),
            ),
            worker_instance_id=mint_worker_instance_id(),
        ),
        WorkerCapabilityRecoveryOutcome(
            phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
            provenance=_provenance(),
        ),
        WorkerCapabilityAcquisitionDecision(
            decision_id="decision-p3-r1",
            worker_instance_id=mint_worker_instance_id(),
            obstacle_id="obs",
            recovery_decision_id="recovery-p3-r1",
            need_id="need-1",
            capability_profile_ref=CapabilityProfileRef(
                profile_id="profile/default",
                version=initial_profile_version(),
            ),
            disposition=CapabilityAcquisitionDisposition.CONFIGURE_EXISTING,
            reason_code=CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED,
            selected_candidate=WorkerCapabilityCandidate(
                candidate_id=derive_worker_capability_candidate_id(
                    candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                    capability_ref="integration:sqlite",
                    configuration_ref=str(_CONFIG_REF),
                ),
                candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                capability_ref="integration:sqlite",
                source_domain="integrations",
                operations=("query",),
                risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
                evidence_refs=(),
                discovered_at=_NOW,
                configuration_ref=str(_CONFIG_REF),
                capability_identity=CapabilityIdentityKey(
                    kind=CapabilityKind.TOOL,
                    source_id="official.marketplace",
                    source_kind=CapabilitySourceKind.OFFICIAL,
                    logical_id="tools.database.relational",
                ),
            ),
            autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
            decided_at=_NOW,
            decision_policy_version="v1",
            evidence_refs=(),
        ),
    ).adoption
    assert adoption is not None
    execution_id = mint_execution_id()
    from intergrax.contracts.capability_qualification.qualified_capability_binding import (
        QualifiedCapabilityExecutionTarget,
    )

    dispatch_request = BoundCapabilityExecutionDispatchRequest(
        execution_request_id=intent.execution_request_id,
        execution_target=build_marketplace_tool_execution_target(
            execution_target_reference=execution_target_reference_for_marketplace_qualified_tool(
                deps.handoff_id,
            ),
            binding_provider_id=MARKETPLACE_TOOL_QUALIFIED_CAPABILITY_BINDING_PROVIDER_ID,
            qualified_subject_reference=intent.qualified_subject_reference,
        ),
        tenant_id=_TENANT,
        task_id=_TASK_ID,
    )
    result = _with_active_execution_identity(
        execution_id,
        lambda: composition.handler.dispatch_once(
            dispatch_request,
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=execution_id,
            integration_configuration_adoption=adoption,
        ),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    assert materialization.count == 1
    assert integration.io_calls == 1
    assert realize.realize.call_count == 1


def test_governed_fulfillment_wiring_accepts_production_inner_dispatch() -> None:
    composition = _production_composition(
        materialization=_CountingMaterialization(_FakeRelationalIntegration(object())),
        catalog_invoker=_governed_database_catalog_invoker(),
        deps=_marketplace_database_deps(),
    )
    dispatch, _ = build_production_marketplace_qualified_capability_execution_dispatch(
        composition=composition,
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    need_repo = InMemoryWorkerRecoveryObstacleCapabilityNeedRepository()
    principal_repo = InMemoryWorkerPrincipalBindingRepository()
    wiring = build_worker_recovery_governed_fulfillment_wiring(
        recovery=MagicMock(),
        direct_reuse=MagicMock(),
        inner_dispatch=dispatch,
        binding=MagicMock(),
        obstacle_capability_need_reader=need_repo,
        task_context_reader=MagicMock(),
        principal_binding_repository=principal_repo,
        capability_profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_PROFILE),
        ),
    )
    assert wiring.governed_dispatch is not None


def test_alternate_provider_materialization_without_composition_edit() -> None:
    token = object()
    alt = _FakeRelationalIntegration(token, provider_id="alt-relational")
    materialization = _CountingMaterialization(alt)
    deps = _marketplace_database_deps()
    composition = build_production_marketplace_configured_execution_composition(
        intent_repository=deps.intent_repo,
        stage_repository=deps.stage_repo,
        activation_read=MagicMock(),
        acquisition=MagicMock(),
        host_profile_id=_HOST,
        material_provider=_DatabaseMaterialProvider(),
        catalog_tool_invoker=_governed_database_catalog_invoker(),
        configuration_pinning_kv_store=InMemoryKVStore(),
        materialization=materialization,
        runtime_event_bus=_production_runtime_event_bus(),
    )
    assert composition.resolution is not None
    assert materialization.instance is alt


def test_configure_existing_e2e_execution_bound_fulfillment() -> None:
    token = object()
    integration = _FakeRelationalIntegration(token)
    materialization = _CountingMaterialization(integration)
    deps = _marketplace_database_deps()
    composition = _production_composition(
        materialization=materialization,
        catalog_invoker=_governed_database_catalog_invoker(),
        deps=deps,
    )
    bound_dispatch, bound_delegate = (
        build_production_marketplace_configured_execution_bound_dispatch(
            composition=composition,
            runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
        )
    )
    worker = mint_worker_instance_id()
    configured_execution = build_production_marketplace_configured_execution_fulfillment(
        intent_repository=deps.intent_repo,
        execution_bound_dispatch=bound_dispatch,
        authority_admission=build_worker_execution_admission_for_uca6c(
            worker_instance_id=worker,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        ),
    )
    binding = ConfiguredCapabilityBinding(
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        configuration_type="test",
        configuration_version="v1",
        configuration_fingerprint="fp",
        realization_evidence_refs=("evidence-1",),
    )
    opportunity = ExistingCapabilityConfigurationOpportunity(
        configuration_ref=_CONFIG_REF,
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="default",
        current_revision="rev-1",
        configuration=MagicMock(
            configuration_type="test",
            configuration_version="v1",
            configuration_fingerprint="fp",
        ),
        configuration_fingerprint="fp",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    read = MagicMock()
    read.read_exact.return_value = opportunity
    realize = MagicMock()
    realize.realize.return_value = ExistingCapabilityConfigurationRealizationResult(
        request_id="req-e2e",
        configured_binding=binding,
        authorization_evidence=MagicMock(),
    )
    principal = MagicMock()
    principal.tenant_id = _TENANT
    principal.principal_id = "principal-1"
    resolver = MagicMock()
    resolver.resolve.return_value = principal
    configured_fulfillment = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=realize,
        principal_binding_resolver=resolver,
    )
    decision = WorkerCapabilityAcquisitionDecision(
        decision_id="decision-e2e",
        worker_instance_id=worker,
        obstacle_id="obs",
        recovery_decision_id="recovery-e2e",
        need_id="need-1",
        capability_profile_ref=CapabilityProfileRef(
            profile_id="profile/default",
            version=initial_profile_version(),
        ),
        disposition=CapabilityAcquisitionDisposition.CONFIGURE_EXISTING,
        reason_code=CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED,
        selected_candidate=WorkerCapabilityCandidate(
            candidate_id=derive_worker_capability_candidate_id(
                candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                capability_ref="integration:sqlite",
                configuration_ref=str(_CONFIG_REF),
            ),
            candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
            capability_ref="integration:sqlite",
            source_domain="integrations",
            operations=("database.query",),
            risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
            evidence_refs=(),
            discovered_at=_NOW,
            configuration_ref=str(_CONFIG_REF),
            capability_identity=CapabilityIdentityKey(
                kind=CapabilityKind.TOOL,
                source_id="official.marketplace",
                source_kind=CapabilitySourceKind.OFFICIAL,
                logical_id="tools.database.relational",
            ),
        ),
        autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        decided_at=_NOW,
        decision_policy_version="v1",
        evidence_refs=(),
    )
    recovery = WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
        provenance=_provenance(),
        worker_acquisition_decision=decision,
    )
    coordinator = WorkerCapabilityFulfillmentCoordinator(
        recovery=MagicMock(coordinate_recovery=MagicMock(return_value=recovery)),
        resume=MagicMock(),
        direct_reuse=MagicMock(),
        configured_fulfillment=configured_fulfillment,
        configured_execution=configured_execution,
    )
    need = MagicMock()
    need.recovery_decision_id = "recovery-e2e"
    need.required_operations = ("database.query",)
    request = WorkerCapabilityFulfillmentRequest(
        worker_instance_id=worker,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
        acquisition_request=MagicMock(need=need),
        requested_at=_NOW,
        requested_authority_scopes=(_READ,),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
    )
    result = coordinator.fulfill(request, decided_at=_NOW)
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    assert bound_delegate.execute_calls == 1
    assert materialization.count == 1
    assert integration.io_calls == 1
