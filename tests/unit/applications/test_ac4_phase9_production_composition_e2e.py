# © Artur Czarnecki. All rights reserved.

"""AC-4 Phase 9 — reference production composition delegated subtask E2E."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    project_package_contract_capabilities,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
    AgentProjectMetadata,
)
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentDiscoveryCandidateIdentity,
    CatalogPackageResolution,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskDelegate,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskNoEligibleAgent,
    DelegatedSubtaskRequest,
    DelegatedSubtaskService,
    DelegatedSubtaskTaskScopeMismatch,
    DelegationId,
    SpecialistInvocationPort,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.dynamic_acquisition import (
    DynamicAgentAcquisitionResult,
)
from intergrax.agent_distribution.federated_discovery import (
    FederatedAgentDiscoveryStrategy,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryDeploymentInstanceStore,
)
from intergrax.agent_distribution.materialization import MaterializationOutput
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_resolution_request,
    build_task_capability_rule,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.production_agent_capability_runtime import (
    AgentCapabilityApplicationComposition,
    ProductionAgentCapabilityRuntime,
    ProductionAgentPlatformAdminConfig,
    build_delegated_subtask_service_factory,
    build_production_agent_capability_runtime,
)
from intergrax.applications._shared.production_delegated_subtask_plans import (
    ProductionDelegatedSubtaskPlanConfig,
    derive_production_delegated_binding_id,
    derive_production_delegated_installation_slot,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleServices,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.core.qualification import QualificationStatus
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.boundary import (
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _install_request,
    admin_test_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app-a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_SOURCE_ID = "builtin-1"
_CATALOG_ENTRY_ID = "cat-researcher"
_PACKAGE_ID = "intergrax-local-search-agent"
_LEGAL_PACKAGE = "legal-agent"
_META_REF = "meta://search"
_DIGEST = "sha256:" + ("a" * 64)
_ARTIFACT = "sha256:" + ("d" * 64)
_BINDING_ID = "bind-search"
_LOGICAL_AGENT = "researcher"


@dataclass(frozen=True)
class OcrRequest:
    document_ref: str


@dataclass(frozen=True)
class OcrResult:
    text: str


class _AllowEvaluator:
    def evaluate(self, request: object) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.ALLOW, reason="test_allow")


class _MetadataProvider:
    def __init__(self, records: dict[str, AgentProjectMetadata]) -> None:
        self._records = records

    def get_metadata(self, metadata_ref: str) -> AgentProjectMetadata | None:
        return self._records.get(metadata_ref)


class _DeterministicMaterializer:
    topology = MaterializationTopology.OCI_IMAGE
    materializer_id = "intergrax.phase9-test"
    materializer_version = "1.0.0"

    def materialize(self, materialization_input: object) -> MaterializationOutput:
        del materialization_input
        return MaterializationOutput(
            materialization_artifact_digest=_ARTIFACT,
            artifact_locator="test://artifact",
            health_check_evidence_ref="test://health",
            runtime_graph_manifest_path=".intergrax-runtime-graph.json",
            topology=self.topology,
        )


class _Phase9Catalog:
    def __init__(self) -> None:
        self._source = CatalogSourceIdentity(
            catalog_source_id=_SOURCE_ID,
            provider_kind=CatalogProviderKind.BUILTIN,
        )
        self._entry = AgentCatalogEntry(
            catalog_entry_id=_CATALOG_ENTRY_ID,
            catalog_source=self._source,
            display_name="Researcher",
            package_id_line=_PACKAGE_ID,
        )
        self._resolution = CatalogPackageResolution(
            entry=self._entry,
            package_candidate=AgentPackageCandidate(
                distribution_package_id=_PACKAGE_ID,
                package_version="1.0.0",
                package_digest=_DIGEST,
            ),
            artifact_locator="catalog://artifact/researcher",
        )

    @property
    def catalog_source_id(self) -> str:
        return self._source.catalog_source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return [self._entry]

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        del entry, version_selector
        return self._resolution

    def health(self) -> None:
        return None


class _FixedTrustFactory:
    def build_trust_record(
        self,
        *,
        package_digest: str,
        package_id: str,
    ) -> AgentInstallationTrustRecord:
        del package_id
        return AgentInstallationTrustRecord(
            qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
            package_digest=package_digest,
            publisher_identity_ref="publisher:acme",
            source_provider_id="builtin",
            trust_evidence_refs=(
                AgentTrustEvidenceRef(
                    evidence_id="evidence:phase9",
                    kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                ),
            ),
        )


def _source() -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )


def _identity(
    package_id: str, *, digest: str = _DIGEST
) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(),
        package=AgentPackageCandidate(
            distribution_package_id=package_id,
            package_version="1.0.0",
            package_digest=digest,
        ),
    )


def _discovery_candidate(
    package_id: str,
    *,
    capability_ids: tuple[str, ...],
) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(package_id),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=_CATALOG_ENTRY_ID,
        artifact_locator=f"catalog://artifact/{package_id}",
    )


def _baseline_resolver():
    return build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.document.ocr.v1",
                task_kind="document.ocr",
                required=("document.ocr",),
            ),
        ),
    )


def _discovery_strategy(
    *candidates: AgentDiscoveryCandidate,
) -> AgentDiscoveryStrategy:
    return FederatedAgentDiscoveryStrategy(
        strategies=(
            StaticAgentDiscoveryStrategy(
                strategy_id=AgentDiscoveryStrategyId(value="static.phase9"),
                candidates=candidates,
            ),
        ),
    )


def _application_composition(
    catalog: _Phase9Catalog,
    *,
    discovery: AgentDiscoveryStrategy | None = None,
) -> AgentCapabilityApplicationComposition:
    metadata = _MetadataProvider(
        {
            _META_REF: AgentProjectMetadata(
                distribution_package_id=_PACKAGE_ID,
                declared_contracts=(
                    AgentPackageContractDeclaration(
                        contract_id="contract.v1",
                        contract_version="1",
                        capabilities=("document.ocr",),
                    ),
                ),
            ),
        },
    )
    return AgentCapabilityApplicationComposition(
        capability_resolver=_baseline_resolver(),
        catalog_providers=(catalog,),
        package_metadata_refs={_PACKAGE_ID: _META_REF},
        package_logical_agents={_PACKAGE_ID: _LOGICAL_AGENT},
        trust_record_factory=_FixedTrustFactory(),
        admin_config=ProductionAgentPlatformAdminConfig(
            metadata_provider=metadata,
            materialization_service=RuntimeMaterializationService(
                {MaterializationTopology.OCI_IMAGE: _DeterministicMaterializer()},
            ),
            dependency_resolver=make_identity_dependency_resolver(),
            mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
                evaluator=_AllowEvaluator(),
            ),
            environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
                "tenant-test",
            ),
            catalog_provider=catalog,
        ),
        delegated_plan_config=ProductionDelegatedSubtaskPlanConfig(
            application_id=_APP,
            application_environment_id=_ENV,
            application_release_id=_RELEASE,
            package_metadata_refs={_PACKAGE_ID: _META_REF},
            package_logical_agents={_PACKAGE_ID: _LOGICAL_AGENT},
        ),
        discovery_strategy=discovery,
    )


def _build_test_lifecycle_services(
    composition: ProductionProcessComposition,
) -> ReferenceProductionLifecycleServices:
    state = composition.agent_platform_runtime.distribution_state
    stores = composition.agent_platform_runtime.stores
    projection_input_store = InMemoryRegistryProjectionInputStore()
    projection_coordinator = FakeRuntimeServingProjectionCoordinator()
    deployment_instance_store = InMemoryDeploymentInstanceStore(state)
    activation_service = ActivationService(
        revision_store=stores.revision_store,
        deployment_instance_store=deployment_instance_store,
        serving_store=stores.serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=projection_coordinator,
    )
    return ReferenceProductionLifecycleServices(
        activation_service=activation_service,
        projection_coordinator=ApplicationRegistryProjectionCoordinator(
            revision_store=stores.revision_store,
            input_store=projection_input_store,
            projection_store=stores.registry_projection_store,
        ),
        revision_service=RuntimeRevisionService(stores.revision_store),
        projection_input_store=projection_input_store,
        deployment_instance_store=deployment_instance_store,
    )


@dataclass
class Phase9Harness:
    composition: ProductionProcessComposition
    capability_runtime: ProductionAgentCapabilityRuntime
    delegated_factory: object
    service: DelegatedSubtaskService[OcrRequest, OcrResult]
    catalog: _Phase9Catalog
    plan_config: ProductionDelegatedSubtaskPlanConfig


class _EchoOcrDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        return OcrResult(text=f"ocr:{request.document_ref}")


@dataclass
class _StaticSpecialistInvocation(SpecialistInvocationPort[OcrRequest, OcrResult]):
    delegate: DelegatedSubtaskDelegate[OcrRequest, OcrResult]

    def resolve_delegate(
        self,
        *,
        lease: object,
        acquisition_result: DynamicAgentAcquisitionResult,
    ) -> DelegatedSubtaskDelegate[OcrRequest, OcrResult]:
        del lease, acquisition_result
        return self.delegate


def build_phase9_harness(
    *,
    candidates: tuple[AgentDiscoveryCandidate, ...],
    specialist_delegate: DelegatedSubtaskDelegate[OcrRequest, OcrResult] | None = None,
) -> Phase9Harness:
    catalog = _Phase9Catalog()
    app_composition = _application_composition(
        catalog,
        discovery=_discovery_strategy(*candidates),
    )
    platform_runtime = build_production_agent_platform_runtime()
    composition = ProductionProcessComposition(
        agent_platform_runtime=platform_runtime,
        agent_capability_runtime=None,
    )
    lifecycle_services = _build_test_lifecycle_services(composition)
    capability_runtime = build_production_agent_capability_runtime(
        agent_platform_runtime=platform_runtime,
        application_composition=app_composition,
        lifecycle_services=lifecycle_services,
    )
    composition = ProductionProcessComposition(
        agent_platform_runtime=platform_runtime,
        agent_capability_runtime=capability_runtime,
    )
    delegated_factory = build_delegated_subtask_service_factory(
        capability_runtime=capability_runtime,
        application_composition=app_composition,
    )
    delegate = specialist_delegate or _EchoOcrDelegate()
    service = delegated_factory.create(
        specialist_invocation=_StaticSpecialistInvocation(delegate=delegate),
    )
    return Phase9Harness(
        composition=composition,
        capability_runtime=capability_runtime,
        delegated_factory=delegated_factory,
        service=service,
        catalog=catalog,
        plan_config=app_composition.delegated_plan_config,
    )


def _delegated_request(
    *,
    task_scope: TaskId,
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
) -> DelegatedSubtaskRequest:
    return DelegatedSubtaskRequest(
        delegation_id=DelegationId(delegation_id),
        task_scope_id=task_scope,
        application_id=_APP,
        application_environment_id=_ENV,
        lease_id=TaskScopedAgentLeaseId(lease_id),
        capability_resolution_request=build_task_capability_resolution_request(
            task_kind="document.ocr",
        ),
    )


def _root_identity() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


async def _register_task_scope(*, task_scope: TaskId, run_id: RunId) -> None:
    await ActiveTaskRegistry.register(
        Task(
            task_id=task_scope,
            tenant_id="tenant-test",
            user_id="user-test",
            message="phase9",
            context=TaskContext(),
        ),
        run_id,
    )


async def _run_delegation(
    harness: Phase9Harness,
    *,
    task_scope: TaskId,
    root: ExecutionIdentityBinding | None = None,
) -> object:
    root = root or _root_identity()
    await _register_task_scope(task_scope=task_scope, run_id=root.run_id)
    ledger = create_execution_budget_ledger(RunBudget())
    captured: list[object] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=ledger,
            )
            try:
                result = await harness.service.execute(
                    _delegated_request(task_scope=task_scope),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-1"))
    assert captured
    return captured[0]


@pytest.fixture(autouse=True)
def _clear_active_task_registry() -> None:
    ActiveTaskRegistry.clear_for_tests()
    yield
    ActiveTaskRegistry.clear_for_tests()


@pytest.mark.asyncio
async def test_production_composition_one_admin_universe() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
    )
    runtime = harness.capability_runtime
    assert runtime.dynamic_acquisition._lifecycle is runtime.admin_service
    assert (
        runtime.task_scoped_agents._release_service._lifecycle is runtime.admin_service
    )
    assert (
        runtime.task_scoped_agents._acquisition_service._acquisition
        is runtime.dynamic_acquisition
    )


@pytest.mark.asyncio
async def test_production_composition_delegated_happy_path() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("legal.analysis",)),
        ),
    )
    task_scope = mint_task_id()
    result = await _run_delegation(harness, task_scope=task_scope)
    assert result.result.text == "ocr:doc-1"
    assert result.selected_identity.package.distribution_package_id == _PACKAGE_ID
    assert result.acquisition_result.resolved_package_identity.package_digest == _DIGEST
    assert (
        result.acquisition_result.traffic_serving_revision_id
        == result.acquisition_result.runtime_revision_id
    )
    lease = harness.capability_runtime.lease_store.get(
        TaskScopedAgentLeaseId("lease-delegate-1")
    )
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASED


@pytest.mark.asyncio
async def test_production_composition_source_qualified_identity_preserved() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
    )
    task_scope = mint_task_id()
    result = await _run_delegation(harness, task_scope=task_scope)
    assert result.selected_identity == result.acquisition_result.selected_identity
    assert result.lease_id == TaskScopedAgentLeaseId("lease-delegate-1")
    assert result.selected_identity.source.catalog_source_id == _SOURCE_ID


@pytest.mark.asyncio
async def test_production_composition_child_execution_lineage() -> None:
    child_execution_id: ExecutionId | None = None
    child_parent_execution_id: ExecutionId | None = None
    child_run_id: RunId | None = None
    child_attempt_id: object | None = None

    class LineageDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            nonlocal \
                child_execution_id, \
                child_parent_execution_id, \
                child_run_id, \
                child_attempt_id
            child_execution_id = require_active_execution_id()
            child_parent_execution_id = peek_active_parent_execution_id()
            identity = peek_active_execution_identity()
            assert identity is not None
            child_run_id, child_attempt_id = identity
            return OcrResult(text=request.document_ref)

    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=LineageDelegate(),
    )
    root = _root_identity()
    task_scope = mint_task_id()
    await _run_delegation(harness, task_scope=task_scope, root=root)
    assert child_execution_id is not None
    assert child_execution_id != root.execution_id
    assert child_parent_execution_id == root.execution_id
    assert child_run_id == root.run_id
    assert child_attempt_id == root.attempt_id


@pytest.mark.asyncio
async def test_production_composition_no_eligible_agent() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("legal.analysis",)),
        ),
    )
    task_scope = mint_task_id()
    root = _root_identity()
    await _register_task_scope(task_scope=task_scope, run_id=root.run_id)

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskNoEligibleAgent):
                await harness.service.execute(
                    _delegated_request(task_scope=task_scope),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="none"))


@pytest.mark.asyncio
async def test_production_composition_cross_task_scope_rejected() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
    )
    active_scope = mint_task_id()
    claimed_scope = mint_task_id()
    root = _root_identity()
    await _register_task_scope(task_scope=active_scope, run_id=root.run_id)

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(DelegatedSubtaskTaskScopeMismatch):
                await harness.service.execute(
                    _delegated_request(task_scope=claimed_scope),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="cross"))


@pytest.mark.asyncio
async def test_production_composition_catalog_provider_coherence() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
    )
    source_ids = harness.capability_runtime.catalog_registry.registered_source_ids
    assert source_ids == (_SOURCE_ID,)
    assert harness.catalog.catalog_source_id in source_ids


@pytest.mark.asyncio
async def test_production_composition_persistent_binding_survives_release() -> None:
    harness = build_phase9_harness(
        candidates=(
            _discovery_candidate(_PACKAGE_ID, capability_ids=("document.ocr",)),
        ),
    )
    admin = harness.capability_runtime.admin_service
    principal = admin_test_principal()
    plan_config = harness.plan_config
    persistent_binding_id = derive_production_delegated_binding_id(
        config=plan_config,
        selected_identity=_identity(_PACKAGE_ID),
        package_id=_PACKAGE_ID,
    )
    persistent_slot_id = derive_production_delegated_installation_slot(
        config=plan_config,
        package_id=_PACKAGE_ID,
    )
    admin.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request().model_copy(
            update={
                "installation_slot_id": persistent_slot_id,
                "installation_id": f"inst-{persistent_slot_id}",
                "artifact_store_ref": f"store://artifacts/{persistent_slot_id}",
            },
        ),
        principal=principal,
    )
    admin.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=BindAgentRequest(
            mutation_id="mut-persistent-bind",
            application_binding_id=persistent_binding_id,
            logical_agent_id=_LOGICAL_AGENT,
            installation_slot_id=persistent_slot_id,
            enablement=True,
        ),
        principal=principal,
    )
    binding = admin.list_bindings(
        application_id=_APP,
        application_environment_id=_ENV,
    ).bindings[0]
    admin.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=persistent_binding_id,
        request=SetAgentEnablementRequest(
            mutation_id="mut-persistent-enable",
            expected_revision=binding.binding_revision,
        ),
        principal=principal,
    )
    task_scope = mint_task_id()
    await _run_delegation(harness, task_scope=task_scope)
    status = admin.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id=_LOGICAL_AGENT,
    )
    assert status.enabled_in_desired_state


def test_create_reference_production_process_composition_without_app_is_ac3_only() -> (
    None
):
    composition = create_reference_production_process_composition()
    assert composition.agent_capability_runtime is None


def test_create_reference_production_process_composition_with_app_wires_capability() -> (
    None
):
    catalog = _Phase9Catalog()
    app = _application_composition(catalog)
    composition = create_reference_production_process_composition(
        application_composition=app,
    )
    assert composition.agent_capability_runtime is not None
    assert (
        composition.agent_capability_runtime.admin_service
        is composition.agent_capability_runtime.dynamic_acquisition._lifecycle
    )
