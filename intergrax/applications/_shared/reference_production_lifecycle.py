# © Artur Czarnecki. All rights reserved.

"""REFERENCE / DEMO lifecycle orchestration — not production deployment authority.

Wires AP-9/AP-10 services from one ``ProductionProcessComposition`` for explicit
single-process proofs. Uses ``FakeInMemoryRuntimeDeploymentAdapter`` and expects the
caller to supply frozen ``RegistryProjectionInputBundle`` inputs.

**Not production deployment/readiness authority.** Real production registry inputs
must be assembled via ``production_registry_projection_input_bundle`` from canonical
lifecycle state and immutable materialization artifacts.

Does **not** read manifest defaults, fabricate revisions at host startup, or own
application business logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agent_distribution.activation import ActivationService
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.control_plane_governance import (
    ApplicationEnvironmentTenantResolver,
    authorize_scoped_control_plane_mutation,
    build_activation_mutation_request,
    build_admit_runtime_revision_mutation_request,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryDeploymentInstanceStore,
)
from intergrax.agent_distribution.sqlite_stores import SqliteAgentDistributionStoreBundle
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.stores import DeploymentInstanceStore
from intergrax.applications._shared.deploy_activate_pipeline import (
    DeployActivatePipelineContext,
    ReferenceProductionLifecycleError,
    ReferenceProductionLifecycleGovernanceBlockedError,
    ReferenceProductionLifecycleResult,
    run_deploy_and_activate,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    MaterializedRegistryProjection,
    RegistryProjectionInputBundle,
    RegistryProjectionInputStore,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    InMemoryRuntimeRegistryProjectionDescriptorStore,
    RuntimeRegistryProjectionDescriptorStore,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class ReferenceProductionLifecycleServices:
    """Lifecycle service bundle wired from one process composition."""

    activation_service: ActivationService
    projection_coordinator: ApplicationRegistryProjectionCoordinator
    revision_service: RuntimeRevisionService
    projection_input_store: RegistryProjectionInputStore
    descriptor_store: RuntimeRegistryProjectionDescriptorStore
    deployment_instance_store: DeploymentInstanceStore


def wire_reference_production_lifecycle_services(
    composition: ProductionProcessComposition,
) -> ReferenceProductionLifecycleServices:
    """Construct AP lifecycle services from the composition's shared distribution state."""
    state = composition.agent_platform_runtime.distribution_state
    stores = composition.agent_platform_runtime.stores
    revision_store = stores.revision_store
    projection_input_store = InMemoryRegistryProjectionInputStore()
    descriptor_store = InMemoryRuntimeRegistryProjectionDescriptorStore()
    projection_coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=projection_input_store,
        projection_store=stores.registry_projection_store,
    )
    deployment_instance_store = InMemoryDeploymentInstanceStore(state)
    activation_service = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_instance_store,
        serving_store=stores.serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=projection_coordinator,
        materialization_store=stores.materialization_store,
    )
    return ReferenceProductionLifecycleServices(
        activation_service=activation_service,
        projection_coordinator=projection_coordinator,
        revision_service=RuntimeRevisionService(revision_store),
        projection_input_store=projection_input_store,
        descriptor_store=descriptor_store,
        deployment_instance_store=deployment_instance_store,
    )


def wire_durable_reference_production_lifecycle_services(
    composition: ProductionProcessComposition,
    *,
    durable_store_bundle: SqliteAgentDistributionStoreBundle,
) -> ReferenceProductionLifecycleServices:
    """Construct AP lifecycle services from durable SQLite store bundles."""
    from intergrax.applications._shared.production_lifecycle import (
        build_descriptor_backed_projection_input_store,
    )

    stores = composition.agent_platform_runtime.stores
    revision_store = durable_store_bundle.revision_store
    projection_input_store = build_descriptor_backed_projection_input_store(
        durable_store_bundle,
    )
    descriptor_store = durable_store_bundle.projection_descriptor_store
    projection_coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=projection_input_store,
        projection_store=stores.registry_projection_store,
    )
    deployment_instance_store = durable_store_bundle.deployment_instance_store
    activation_service = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_instance_store,
        serving_store=durable_store_bundle.serving_store,
        activation_store=durable_store_bundle.activation_store,
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=projection_coordinator,
        materialization_store=durable_store_bundle.materialization_store,
    )
    return ReferenceProductionLifecycleServices(
        activation_service=activation_service,
        projection_coordinator=projection_coordinator,
        revision_service=RuntimeRevisionService(revision_store),
        projection_input_store=projection_input_store,
        descriptor_store=descriptor_store,
        deployment_instance_store=deployment_instance_store,
    )


class ReferenceProductionLifecycleLauncher:
    """Explicit deploy → project → activate for one reference production process."""

    def __init__(
        self,
        composition: ProductionProcessComposition,
        *,
        services: ReferenceProductionLifecycleServices | None = None,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary
        | None = None,
        environment_tenant_resolver: ApplicationEnvironmentTenantResolver | None = None,
    ) -> None:
        self._composition = composition
        self._services = services or wire_reference_production_lifecycle_services(
            composition
        )
        self._mutation_authorization_boundary = mutation_authorization_boundary
        self._environment_tenant_resolver = environment_tenant_resolver

    @property
    def process_composition(self) -> ProductionProcessComposition:
        return self._composition

    @property
    def services(self) -> ReferenceProductionLifecycleServices:
        return self._services

    def deploy_and_activate(
        self,
        projection_input: RegistryProjectionInputBundle,
        activation_request: ActivateRuntimeRevisionRequest,
        *,
        principal: RequestIdentity,
        admission_mutation_id: str,
    ) -> ReferenceProductionLifecycleResult:
        """Prepare registry projection and commit activation for one explicit revision."""
        boundary = self._require_mutation_authorization_boundary()
        resolver = self._require_environment_tenant_resolver()
        return run_deploy_and_activate(
            DeployActivatePipelineContext(
                composition=self._composition,
                services=self._services,
                mutation_authorization_boundary=boundary,
                environment_tenant_resolver=resolver,
            ),
            projection_input,
            activation_request,
            principal=principal,
            admission_mutation_id=admission_mutation_id,
        )

    def _require_mutation_authorization_boundary(
        self,
    ) -> ControlPlaneMutationAuthorizationBoundary:
        boundary = self._mutation_authorization_boundary
        if boundary is None:
            raise ReferenceProductionLifecycleGovernanceBlockedError(
                "reference production activation requires ControlPlaneMutationAuthorizationBoundary",
                policy_action=PolicyAction.DENY.value,
            )
        return boundary

    def _require_environment_tenant_resolver(
        self,
    ) -> ApplicationEnvironmentTenantResolver:
        resolver = self._environment_tenant_resolver
        if resolver is None:
            raise ReferenceProductionLifecycleGovernanceBlockedError(
                "reference production activation requires ApplicationEnvironmentTenantResolver",
                policy_action=PolicyAction.DENY.value,
            )
        return resolver

    def _authorize_admission(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        mutation_id: str,
        runtime_revision_id: str,
        identity_digest: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        request = build_admit_runtime_revision_mutation_request(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=mutation_id,
            runtime_revision_id=runtime_revision_id,
            identity_digest=identity_digest,
        )
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
        )

    def _authorize_activation(
        self,
        *,
        principal: RequestIdentity,
        application_id: str,
        application_environment_id: str,
        mutation_id: str,
        current_traffic_revision_id: str | None,
        current_serving_pointer_revision: int,
        target_runtime_revision_id: str,
    ) -> ControlPlaneMutationAuthorizationResult:
        resolver = self._require_environment_tenant_resolver()
        boundary = self._require_mutation_authorization_boundary()
        request = build_activation_mutation_request(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=mutation_id,
            current_traffic_revision_id=current_traffic_revision_id,
            current_serving_pointer_revision=current_serving_pointer_revision,
            target_runtime_revision_id=target_runtime_revision_id,
        )
        return self._enforce_authorization_result(
            authorize_scoped_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=resolver,
                request=request,
            ),
        )

    @staticmethod
    def _enforce_authorization_result(
        result: ControlPlaneMutationAuthorizationResult,
    ) -> ControlPlaneMutationAuthorizationResult:
        if result.permitted:
            return result
        action = result.decision.action
        if action is PolicyAction.REQUIRE_HUMAN:
            raise ReferenceProductionLifecycleGovernanceBlockedError(
                "reference production activation requires governed human approval",
                policy_action=action.value,
                authorization_evidence=result.evidence,
                authorization_scope=result.authorization_scope,
            )
        if action is PolicyAction.ESCALATE:
            raise ReferenceProductionLifecycleGovernanceBlockedError(
                "reference production activation requires escalation",
                policy_action=action.value,
                authorization_evidence=result.evidence,
                authorization_scope=result.authorization_scope,
            )
        raise ReferenceProductionLifecycleGovernanceBlockedError(
            "reference production activation denied by control-plane governance",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )


__all__ = [
    "ReferenceProductionLifecycleError",
    "ReferenceProductionLifecycleGovernanceBlockedError",
    "ReferenceProductionLifecycleLauncher",
    "ReferenceProductionLifecycleResult",
    "ReferenceProductionLifecycleServices",
    "wire_durable_reference_production_lifecycle_services",
    "wire_reference_production_lifecycle_services",
]
