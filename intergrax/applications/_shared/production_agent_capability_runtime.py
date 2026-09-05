# © Artur Czarnecki. All rights reserved.

"""Reference production V1 AC-4 capability runtime composition (Phase 9).

Wires Phases 6–8 orchestration into one immutable bundle owned by
``ProductionProcessComposition``. Dynamic acquisition and task-scoped leases
share the same ``AgentPlatformAdminService`` lifecycle universe as AP-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
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
from intergrax.agent_distribution.resolver import DependencyResolver
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionService,
)
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.task_capability_resolution import (
    TaskCapabilityResolver,
)
from intergrax.agent_distribution.task_scoped_agents import (
    InMemoryTaskScopedAgentLeaseStore,
    TaskScopedAgentLeaseStore,
    TaskScopedAgentService,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    ProductionAgentPlatformRuntime,
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
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.delegated_subtask_child_port import (
    as_child_execution_port,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.task.active_task_registry import (
    ActiveTaskRegistryTaskScopeResolver,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ProductionAgentPlatformAdminConfig:
    """Explicit admin dependencies for one reference production process."""

    metadata_provider: AgentProjectMetadataProvider
    materialization_service: RuntimeMaterializationService
    dependency_resolver: DependencyResolver
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary
    environment_tenant_resolver: ApplicationEnvironmentTenantResolver
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
    lifecycle_services: ReferenceProductionLifecycleServices


@dataclass(frozen=True, slots=True)
class DelegatedSubtaskServiceFactory:
    """Typed delegated subtask builder — avoids storing ``DelegatedSubtaskService[Any, Any]``."""

    capability_runtime: ProductionAgentCapabilityRuntime
    application_composition: AgentCapabilityApplicationComposition

    def create(
        self,
        *,
        specialist_invocation: SpecialistInvocationPort[RequestT, ResultT],
        child_execution: ChildExecutionPort[RequestT, ResultT] | None = None,
    ) -> DelegatedSubtaskService[RequestT, ResultT]:
        child_port = (
            child_execution
            if child_execution is not None
            else as_child_execution_port(ChildExecutionRunner[RequestT, ResultT]())
        )
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
            child_execution=child_port,
        )


def build_production_agent_platform_admin_service(
    *,
    agent_platform_runtime: ProductionAgentPlatformRuntime,
    lifecycle_services: ReferenceProductionLifecycleServices,
    admin_config: ProductionAgentPlatformAdminConfig,
) -> AgentPlatformAdminService:
    """Construct one admin facade over the canonical process store universe."""
    state = agent_platform_runtime.distribution_state
    stores = agent_platform_runtime.stores
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    graph_builder = admin_config.graph_builder or CandidateRuntimeGraphBuilder(
        admin_config.metadata_provider,
    )
    return AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=stores.revision_store,
        serving_store=stores.serving_store,
        deployment_instance_store=lifecycle_services.activation_service._deployment_instance_store,  # noqa: SLF001
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
    )
    catalog_registry = _build_catalog_registry(
        application_composition.catalog_providers
    )
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
    selector = (
        application_composition.selection_strategy
        or DeterministicIdentitySelectionStrategy()
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
        lifecycle_services=lifecycle_services,
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
    "ProductionAgentPlatformAdminConfig",
    "build_delegated_subtask_service_factory",
    "build_production_agent_capability_runtime",
    "build_production_agent_platform_admin_service",
]
