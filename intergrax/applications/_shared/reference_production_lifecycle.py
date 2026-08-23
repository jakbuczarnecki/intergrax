# © Artur Czarnecki. All rights reserved.

"""Reference single-process production lifecycle launcher (AGENT-CONSOLIDATION-3-FIX-4).

Wires canonical AP-9/AP-10 services from one ``ProductionProcessComposition``,
consumes explicit ``RegistryProjectionInputBundle`` + ``ActivateRuntimeRevisionRequest``,
and commits traffic-serving state to the same store instances used by host serving.

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
    runtime_revision_admission_identity_digest,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryDeploymentInstanceStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
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


@dataclass(frozen=True, slots=True)
class ReferenceProductionLifecycleResult:
    """Evidence after successful deploy/project/activate on one environment."""

    runtime_revision_id: str
    application_id: str
    application_environment_id: str
    serving_pointer_revision: int
    resolved_projection: MaterializedRegistryProjection


class ReferenceProductionLifecycleError(ValueError):
    """Reference lifecycle launcher input or precondition violation."""


class ReferenceProductionLifecycleGovernanceBlockedError(ReferenceProductionLifecycleError):
    """Control-plane governance blocked reference production activation."""

    def __init__(
        self,
        message: str,
        *,
        policy_action: str,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None,
        authorization_scope: ControlPlaneMutationAuthorizationScope | None = None,
    ) -> None:
        super().__init__(message)
        self.policy_action = policy_action
        self.authorization_evidence = authorization_evidence
        self.authorization_scope = authorization_scope


def wire_reference_production_lifecycle_services(
    composition: ProductionProcessComposition,
) -> ReferenceProductionLifecycleServices:
    """Construct AP lifecycle services from the composition's shared distribution state."""
    state = composition.agent_platform_runtime.distribution_state
    stores = composition.agent_platform_runtime.stores
    revision_store = InMemoryRuntimeRevisionStore(state)
    projection_input_store = InMemoryRegistryProjectionInputStore()
    projection_coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=revision_store,
        input_store=projection_input_store,
        projection_store=stores.registry_projection_store,
    )
    activation_service = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=InMemoryDeploymentInstanceStore(state),
        serving_store=stores.serving_store,
        activation_store=InMemoryApplicationEnvironmentActivationStore(state),
        deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
        projection_coordinator=projection_coordinator,
    )
    return ReferenceProductionLifecycleServices(
        activation_service=activation_service,
        projection_coordinator=projection_coordinator,
        revision_service=RuntimeRevisionService(revision_store),
        projection_input_store=projection_input_store,
    )


class ReferenceProductionLifecycleLauncher:
    """Explicit deploy → project → activate for one reference production process."""

    def __init__(
        self,
        composition: ProductionProcessComposition,
        *,
        services: ReferenceProductionLifecycleServices | None = None,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
        environment_tenant_resolver: ApplicationEnvironmentTenantResolver | None = None,
    ) -> None:
        self._composition = composition
        self._services = services or wire_reference_production_lifecycle_services(composition)
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
        revision = projection_input.runtime_revision
        application_id = revision.application_id
        application_environment_id = revision.application_environment_id
        runtime_revision_id = revision.runtime_revision_id
        normalized_admission_mutation_id = admission_mutation_id.strip()
        if not normalized_admission_mutation_id:
            raise ReferenceProductionLifecycleError(
                "reference production admission requires explicit admission_mutation_id"
            )

        if activation_request.runtime_revision_id != runtime_revision_id:
            raise ReferenceProductionLifecycleError(
                "activation request runtime_revision_id does not match projection input"
            )
        artifact_digest = projection_input.materialization_artifact_digest
        if artifact_digest is None:
            raise ReferenceProductionLifecycleError(
                "projection input requires materialization_artifact_digest"
            )
        if activation_request.expected_artifact_digest != artifact_digest:
            raise ReferenceProductionLifecycleError(
                "activation expected_artifact_digest does not match projection input"
            )
        if revision.materialization_artifact_digest is not None:
            if revision.materialization_artifact_digest != artifact_digest:
                raise ReferenceProductionLifecycleError(
                    "runtime revision materialization_artifact_digest mismatch with projection input"
                )
        if revision.materialized_runtime_lock_digest is None:
            raise ReferenceProductionLifecycleError(
                "runtime revision requires materialized_runtime_lock_digest for admission"
            )
        if revision.runtime_graph_digest is None:
            raise ReferenceProductionLifecycleError(
                "runtime revision requires runtime_graph_digest for admission"
            )
        if revision.materialization_topology is None:
            raise ReferenceProductionLifecycleError(
                "runtime revision requires materialization_topology for admission"
            )

        self._require_environment_tenant_resolver()
        self._require_mutation_authorization_boundary()

        admission_identity_digest = runtime_revision_admission_identity_digest(
            runtime_revision_id=runtime_revision_id,
            application_release_id=revision.application_release_id,
            platform_version=revision.platform_version,
            effective_roster_revision_id=revision.effective_roster_revision_id,
            lock_digest=revision.materialized_runtime_lock_digest,
            graph_digest=revision.runtime_graph_digest,
            materialization_topology=revision.materialization_topology.value,
            materialization_artifact_digest=artifact_digest,
            build_input_digest=revision.build_input_digest,
        )
        self._authorize_admission(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=normalized_admission_mutation_id,
            runtime_revision_id=runtime_revision_id,
            identity_digest=admission_identity_digest,
        )

        revision_service = self._services.revision_service
        candidate = revision.model_copy(
            update={
                "revision_state": RuntimeRevisionState.CANDIDATE,
                "materialization_artifact_digest": artifact_digest,
            }
        )
        revision_service.persist_candidate_revision(candidate)
        validated = candidate.model_copy(
            update={"revision_state": RuntimeRevisionState.VALIDATED}
        )
        revision_service.mark_validated(runtime_revision_id, validated_revision=validated)

        self._services.projection_input_store.register(projection_input)

        activation = self._services.activation_service
        activation.prepare_candidate(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=runtime_revision_id,
            artifact_locator=activation_request.artifact_locator,
        )
        serving_store = self._composition.agent_platform_runtime.stores.serving_store
        serving = serving_store.get_serving_record(application_id, application_environment_id)
        current_traffic_revision_id = (
            serving.traffic_serving_revision_id if serving is not None else None
        )
        current_pointer_revision = serving.serving_pointer_revision if serving is not None else 0
        self._authorize_activation(
            principal=principal,
            application_id=application_id,
            application_environment_id=application_environment_id,
            mutation_id=activation_request.mutation_id,
            current_traffic_revision_id=current_traffic_revision_id,
            current_serving_pointer_revision=current_pointer_revision,
            target_runtime_revision_id=runtime_revision_id,
        )
        committed = activation.commit_activation(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=runtime_revision_id,
            expected_prior_traffic_revision_id=activation_request.expected_prior_traffic_revision_id,
            expected_serving_pointer_revision=activation_request.expected_serving_pointer_revision,
            expected_artifact_digest=activation_request.expected_artifact_digest,
        )

        resolved = bootstrap_production_registry_projection(
            application_id=application_id,
            application_environment_id=application_environment_id,
            stores=self._composition.agent_platform_runtime.stores,
        )
        if resolved.evidence.runtime_revision_id != runtime_revision_id:
            raise ReferenceProductionLifecycleError(
                "serving authority resolved unexpected runtime revision"
            )

        serving_record = committed.value.serving_record
        return ReferenceProductionLifecycleResult(
            runtime_revision_id=runtime_revision_id,
            application_id=application_id,
            application_environment_id=application_environment_id,
            serving_pointer_revision=serving_record.serving_pointer_revision,
            resolved_projection=resolved,
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
    "wire_reference_production_lifecycle_services",
]
