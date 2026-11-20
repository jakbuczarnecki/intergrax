# © Artur Czarnecki. All rights reserved.

"""Reference production V1 AC-4 capability runtime composition (Phase 9).

Wires Phases 6–8 orchestration into one immutable bundle owned by
``ProductionProcessComposition``. Dynamic acquisition and task-scoped leases
share the same ``AgentPlatformAdminService`` lifecycle universe as AP-3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from intergrax.agent_distribution.admin_service import (
    AgentPlatformAdminService,
    PackageAttestationEnforcement,
)
from intergrax.agent_distribution.agent_discovery import AgentDiscoveryStrategy
from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadataProvider,
)
from intergrax.agent_distribution.agent_selection import (
    AgentSelectionStrategy,
    DeterministicIdentitySelectionStrategy,
)
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.capability_matching import CapabilityMatcher
from intergrax.agent_distribution.catalog import CatalogSourceProvider
from intergrax.agent_distribution.catalog_discovery import (
    CatalogSourceProviderDiscoveryStrategy,
    EmptyProductionDiscoveryStrategy,
)
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.delegated_subtasks import (
    ChildExecutionPort,
    DelegatedSubtaskService,
    SpecialistInvocationPort,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionService,
)
from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.events import AgentDistributionEvent
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.package_attestation import AgentPackageAttestationVerifier
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.resolver import DependencyResolver
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.sqlite_stores import SqliteAgentDistributionStoreBundle
from intergrax.agent_distribution.task_capability_resolution import (
    TaskCapabilityResolver,
)
from intergrax.agent_distribution.task_scoped_agents import (
    InMemoryTaskScopedAgentLeaseStore,
    TaskScopedAgentLeaseStore,
    TaskScopedAgentService,
)
from intergrax.agent_distribution.trust import (
    AgentPackageTrustPolicy,
    AgentPackageTrustRevocationState,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    ProductionAgentPlatformRuntime,
)
from intergrax.applications._shared.production_delegated_subtask_child_execution_wiring import (
    ProductionDelegatedSubtaskChildExecutionPort,
    build_production_delegated_subtask_child_execution_port,
)
from intergrax.applications._shared.production_delegated_subtask_plans import (
    DelegatedSubtaskTrustRecordFactory,
    ProductionDelegatedSubtaskAcquisitionPlanFactory,
    ProductionDelegatedSubtaskPlanConfig,
    ProductionDelegatedSubtaskReleasePlanFactory,
)
from intergrax.applications._shared.reference_production_lifecycle import (
    ReferenceProductionLifecycleServices,
    wire_reference_production_lifecycle_services,
)
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.physical_delegation_governance import PhysicalDelegationGovernancePort
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.physical_delegation_governance import (
    AllowingPhysicalDelegationGovernance,
)
from intergrax.runtime.security.security_audit_trail import (
    SecurityAuditSink,
    SecurityAuditTrail,
)
from intergrax.runtime.task.active_task_registry import (
    ActiveTaskRegistryTaskScopeResolver,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

#: ``actor_id`` of the security audit entries written by the lifecycle evidence consumer.
LIFECYCLE_EVIDENCE_ACTOR_ID = "agent_distribution.lifecycle"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProductionPackageTrustComposition:
    """Package trust sources the production admin factory composes (T-IGX-051-N Part B, EX-D1-02).

    Every member is required and keyword-only; ``None`` for any member is a ``ValueError`` naming it.
    ``is_lab`` has no default: a lab composition (``lab_package_trust_composition``) has an empty
    revocation state and a policy that admits unsigned packages, and the strict startup refuses it. A
    non-lab composition whose policy (``policy_source()``, evaluated once here) does not set
    ``forbid_unsigned_or_unqualified`` is refused.

    Limit: the strength of ``attestation_verifier`` cannot be checked from the outside. A verifier
    without keys fails every attestation item (unknown key), so under a non-lab policy it refuses
    rather than admits.
    """

    attestation_verifier: AgentPackageAttestationVerifier
    revocation_state_source: Callable[[], AgentPackageTrustRevocationState]
    policy_source: Callable[[], AgentPackageTrustPolicy]
    evaluation_time_source: Callable[[], datetime]
    lifecycle_evidence_trail: SecurityAuditSink
    is_lab: bool

    def __post_init__(self) -> None:
        presence: tuple[tuple[str, bool], ...] = (
            ("attestation_verifier", self.attestation_verifier is not None),
            ("revocation_state_source", self.revocation_state_source is not None),
            ("policy_source", self.policy_source is not None),
            ("evaluation_time_source", self.evaluation_time_source is not None),
            ("lifecycle_evidence_trail", self.lifecycle_evidence_trail is not None),
            ("is_lab", self.is_lab is not None),
        )
        for name, present in presence:
            if not present:
                raise ValueError(f"package_trust composition requires {name}")
        if self.is_lab:
            return
        policy = self.policy_source()
        if not policy.forbid_unsigned_or_unqualified:
            raise ValueError(
                "a non-lab package_trust composition requires a policy with forbid_unsigned_or_unqualified"
            )


def _empty_revocation_state() -> AgentPackageTrustRevocationState:
    return AgentPackageTrustRevocationState()


def _lab_trust_policy() -> AgentPackageTrustPolicy:
    return AgentPackageTrustPolicy(forbid_unsigned_or_unqualified=False)


def _system_clock() -> datetime:
    return datetime.now(UTC)


def lab_package_trust_composition() -> ProductionPackageTrustComposition:
    """Lab and test trust composition: keyless verifier, empty revocation state, unsigned admitted.

    The policy's ``forbid_unsigned_or_unqualified=False`` is read only by the install/upgrade
    attestation step; the trail is the in-memory ``SecurityAuditTrail``. Never a strict root.
    """
    return ProductionPackageTrustComposition(
        attestation_verifier=Ed25519PackageAttestationVerifier(),
        revocation_state_source=_empty_revocation_state,
        policy_source=_lab_trust_policy,
        evaluation_time_source=_system_clock,
        lifecycle_evidence_trail=SecurityAuditTrail(),
        is_lab=True,
    )


class _LifecycleEvidenceSink:
    """Production consumer of the admin lifecycle events (T-IGX-051-N Part B, PDC-D1-R017).

    Writes one security audit entry per event to the trail of the tenant that owns the event's
    application environment. The admin service adds ``application_id`` and
    ``application_environment_id`` to every event it passes here; an event without them cannot be
    attributed to a tenant and is refused (``ValueError``; the admin service surfaces it under its
    sink failure contract).
    """

    def __init__(
        self,
        trail: SecurityAuditSink,
        environment_tenant_resolver: ApplicationEnvironmentTenantResolver,
    ) -> None:
        self._trail = trail
        self._environment_tenant_resolver = environment_tenant_resolver

    def record_events(self, events: tuple[AgentDistributionEvent, ...]) -> None:
        for event in events:
            application_id = event.attributes.get("application_id", "")
            application_environment_id = event.attributes.get("application_environment_id", "")
            if not application_id or not application_environment_id:
                raise ValueError("lifecycle event lacks its application environment; no owning tenant")
            owner = self._environment_tenant_resolver.resolve_tenant_id(
                application_id=application_id,
                application_environment_id=application_environment_id,
            )
            self._trail.append(
                tenant_id=owner,
                action=event.event_type,
                actor_id=LIFECYCLE_EVIDENCE_ACTOR_ID,
                resource="installation:" + event.aggregate_id,
                metadata=dict(event.attributes),
            )


@dataclass(frozen=True, slots=True)
class ProductionAgentPlatformAdminConfig:
    """Explicit admin dependencies for one reference production process.

    ``package_trust`` is required (T-IGX-051-N Part B): the factory composes the trust coordinator,
    the attestation enforcement, the revocation, policy and time sources and the lifecycle evidence
    consumer from it.
    """

    metadata_provider: AgentProjectMetadataProvider
    materialization_service: RuntimeMaterializationService
    dependency_resolver: DependencyResolver
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary
    environment_tenant_resolver: ApplicationEnvironmentTenantResolver
    package_trust: ProductionPackageTrustComposition
    graph_builder: CandidateRuntimeGraphBuilder | None = None
    catalog_provider: CatalogSourceProvider | None = None


@dataclass(frozen=True, slots=True)
class AgentCapabilityApplicationComposition:
    """Application/domain policy injected at explicit production composition boundaries."""

    capability_resolver: TaskCapabilityResolver
    catalog_providers: tuple[CatalogSourceProvider, ...]
    package_metadata_refs: dict[str, str]
    package_logical_agents: dict[str, str]
    trust_record_factory: DelegatedSubtaskTrustRecordFactory
    admin_config: ProductionAgentPlatformAdminConfig
    delegated_plan_config: ProductionDelegatedSubtaskPlanConfig
    discovery_strategy: AgentDiscoveryStrategy | None = None
    selection_strategy: AgentSelectionStrategy | None = None


@dataclass(frozen=True, slots=True)
class ProductionAgentCapabilityRuntime:
    """Process-lifetime AC-4 orchestration bundle — not a domain service."""

    admin_service: AgentPlatformAdminService
    dynamic_acquisition: DynamicAgentAcquisitionService
    task_scoped_agents: TaskScopedAgentService
    lease_store: TaskScopedAgentLeaseStore
    task_scope_authority: ActiveExecutionTaskScopePort
    catalog_registry: CatalogSourceProviderRegistry
    discovery: AgentDiscoveryStrategy
    matcher: CapabilityMatcher
    selector: AgentSelectionStrategy
    acquisition_plan_factory: ProductionDelegatedSubtaskAcquisitionPlanFactory
    release_plan_factory: ProductionDelegatedSubtaskReleasePlanFactory
    physical_delegation_governance: PhysicalDelegationGovernancePort
    lifecycle_services: ReferenceProductionLifecycleServices
    delegated_subtask_child_execution: ProductionDelegatedSubtaskChildExecutionPort


@dataclass(frozen=True, slots=True)
class DelegatedSubtaskServiceFactory:
    """Typed delegated subtask builder — avoids storing ``DelegatedSubtaskService[Any, Any]``."""

    capability_runtime: ProductionAgentCapabilityRuntime
    application_composition: AgentCapabilityApplicationComposition

    def create(
        self,
        *,
        specialist_invocation: SpecialistInvocationPort[RequestT, ResultT],
        child_execution: ChildExecutionPort[RequestT, ResultT],
    ) -> DelegatedSubtaskService[RequestT, ResultT]:
        runtime = self.capability_runtime
        app = self.application_composition
        return DelegatedSubtaskService(
            capability_resolver=app.capability_resolver,
            discovery=runtime.discovery,
            matcher=runtime.matcher,
            selector=runtime.selector,
            task_scoped_agents=runtime.task_scoped_agents,
            task_scope_authority=runtime.task_scope_authority,
            acquisition_plan_factory=runtime.acquisition_plan_factory,
            release_plan_factory=runtime.release_plan_factory,
            specialist_invocation=specialist_invocation,
            child_execution=child_execution,
            physical_delegation_governance=runtime.physical_delegation_governance,
        )


def build_production_agent_platform_admin_service(
    *,
    agent_platform_runtime: ProductionAgentPlatformRuntime,
    lifecycle_services: ReferenceProductionLifecycleServices,
    admin_config: ProductionAgentPlatformAdminConfig,
    durable_store_bundle: SqliteAgentDistributionStoreBundle | None = None,
) -> AgentPlatformAdminService:
    """Construct one admin facade over the canonical process store universe.

    The trust sources, the attestation enforcement and the lifecycle evidence consumer always come
    from ``admin_config.package_trust``.
    """
    state = agent_platform_runtime.distribution_state
    stores = agent_platform_runtime.stores
    if durable_store_bundle is not None:
        installation_store = durable_store_bundle.installation_store
        binding_store = durable_store_bundle.binding_store
        artifact_store = durable_store_bundle.artifact_metadata_store
    else:
        installation_store = InMemoryAgentInstallationStore(state)
        binding_store = InMemoryApplicationAgentBindingStore(state)
        artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    graph_builder = admin_config.graph_builder or CandidateRuntimeGraphBuilder(
        admin_config.metadata_provider,
    )
    trust = admin_config.package_trust
    evidence_sink = _LifecycleEvidenceSink(
        trust.lifecycle_evidence_trail,
        admin_config.environment_tenant_resolver,
    )
    return AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=stores.revision_store,
        serving_store=stores.serving_store,
        deployment_instance_store=lifecycle_services.deployment_instance_store,
        lock_store=stores.lock_store,
        materialization_store=stores.materialization_store,
        effective_roster_snapshot_store=stores.effective_roster_snapshot_store,
        effective_roster_authority=agent_platform_runtime.effective_roster_authority,
        artifact_metadata_store=artifact_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=RuntimeRevisionService(stores.revision_store),
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(artifact_store),
        activation_service=lifecycle_services.activation_service,
        graph_builder=graph_builder,
        materialization_service=admin_config.materialization_service,
        metadata_provider=admin_config.metadata_provider,
        catalog_provider=admin_config.catalog_provider,
        dependency_resolver=admin_config.dependency_resolver,
        mutation_authorization_boundary=admin_config.mutation_authorization_boundary,
        environment_tenant_resolver=admin_config.environment_tenant_resolver,
        package_trust_coordinator=AgentPackageTrustCoordinator(
            attestation_verifier=trust.attestation_verifier,
        ),
        package_trust_revocation_state_source=trust.revocation_state_source,
        package_trust_policy_source=trust.policy_source,
        package_trust_evaluation_time_source=trust.evaluation_time_source,
        package_attestation_enforcement=PackageAttestationEnforcement(
            verifier=trust.attestation_verifier,
        ),
        lifecycle_event_sink=evidence_sink.record_events,
    )


def _build_catalog_registry(
    providers: tuple[CatalogSourceProvider, ...],
) -> CatalogSourceProviderRegistry:
    registry_map = {provider.catalog_source_id: provider for provider in providers}
    return CatalogSourceProviderRegistry(registry_map)


def _resolve_discovery_strategy(
    *,
    application_composition: AgentCapabilityApplicationComposition,
    catalog_registry: CatalogSourceProviderRegistry,
) -> AgentDiscoveryStrategy:
    if application_composition.discovery_strategy is not None:
        return application_composition.discovery_strategy
    if not application_composition.catalog_providers:
        return EmptyProductionDiscoveryStrategy()
    return CatalogSourceProviderDiscoveryStrategy(
        catalog_registry=catalog_registry,
        metadata_provider=application_composition.admin_config.metadata_provider,
        package_metadata_refs=application_composition.package_metadata_refs,
    )


def build_production_agent_capability_runtime(
    *,
    agent_platform_runtime: ProductionAgentPlatformRuntime,
    application_composition: AgentCapabilityApplicationComposition,
    lifecycle_services: ReferenceProductionLifecycleServices | None = None,
    durable_store_bundle: SqliteAgentDistributionStoreBundle | None = None,
    delegated_subtask_child_execution: ProductionDelegatedSubtaskChildExecutionPort | None = None,
) -> ProductionAgentCapabilityRuntime:
    """Wire AC-4 Phases 6–8 from one canonical AP-3 runtime — no duplicate stores."""
    if lifecycle_services is None:
        from intergrax.applications._shared.production_process_composition import (
            ProductionProcessComposition,
        )

        lifecycle_services = wire_reference_production_lifecycle_services(
            ProductionProcessComposition(
                agent_platform_runtime=agent_platform_runtime,
                agent_capability_runtime=None,
            ),
        )
    admin_service = build_production_agent_platform_admin_service(
        agent_platform_runtime=agent_platform_runtime,
        lifecycle_services=lifecycle_services,
        admin_config=application_composition.admin_config,
        durable_store_bundle=durable_store_bundle,
    )
    catalog_registry = _build_catalog_registry(application_composition.catalog_providers)
    dynamic_acquisition = DynamicAgentAcquisitionService(
        catalog_registry=catalog_registry,
        lifecycle=admin_service,
    )
    lease_store: TaskScopedAgentLeaseStore = InMemoryTaskScopedAgentLeaseStore()
    task_scoped_agents = TaskScopedAgentService(
        acquisition=dynamic_acquisition,
        lifecycle=admin_service,
        lease_store=lease_store,
    )
    plan_config = application_composition.delegated_plan_config
    acquisition_plan_factory = ProductionDelegatedSubtaskAcquisitionPlanFactory(
        admin_service=admin_service,
        config=plan_config,
        trust_record_factory=application_composition.trust_record_factory,
    )
    release_plan_factory = ProductionDelegatedSubtaskReleasePlanFactory(
        admin_service=admin_service,
        config=plan_config,
    )
    selector = application_composition.selection_strategy or DeterministicIdentitySelectionStrategy()
    resolved_child_execution = (
        delegated_subtask_child_execution or build_production_delegated_subtask_child_execution_port()
    )
    return ProductionAgentCapabilityRuntime(
        admin_service=admin_service,
        dynamic_acquisition=dynamic_acquisition,
        task_scoped_agents=task_scoped_agents,
        lease_store=lease_store,
        task_scope_authority=ActiveTaskRegistryTaskScopeResolver(),
        catalog_registry=catalog_registry,
        discovery=_resolve_discovery_strategy(
            application_composition=application_composition,
            catalog_registry=catalog_registry,
        ),
        matcher=CapabilityMatcher(),
        selector=selector,
        acquisition_plan_factory=acquisition_plan_factory,
        release_plan_factory=release_plan_factory,
        physical_delegation_governance=AllowingPhysicalDelegationGovernance(),
        lifecycle_services=lifecycle_services,
        delegated_subtask_child_execution=resolved_child_execution,
    )


def build_delegated_subtask_service_factory(
    *,
    capability_runtime: ProductionAgentCapabilityRuntime,
    application_composition: AgentCapabilityApplicationComposition,
) -> DelegatedSubtaskServiceFactory:
    return DelegatedSubtaskServiceFactory(
        capability_runtime=capability_runtime,
        application_composition=application_composition,
    )


__all__ = [
    "AgentCapabilityApplicationComposition",
    "DelegatedSubtaskServiceFactory",
    "ProductionAgentCapabilityRuntime",
    "LIFECYCLE_EVIDENCE_ACTOR_ID",
    "ProductionAgentPlatformAdminConfig",
    "ProductionPackageTrustComposition",
    "build_delegated_subtask_service_factory",
    "build_production_agent_capability_runtime",
    "build_production_agent_platform_admin_service",
    "lab_package_trust_composition",
]
