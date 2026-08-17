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
    ) -> None:
        self._composition = composition
        self._services = services or wire_reference_production_lifecycle_services(composition)

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
    ) -> ReferenceProductionLifecycleResult:
        """Prepare registry projection and commit activation for one explicit revision."""
        revision = projection_input.runtime_revision
        application_id = revision.application_id
        application_environment_id = revision.application_environment_id
        runtime_revision_id = revision.runtime_revision_id

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

        revision_service = self._services.revision_service
        candidate = revision.model_copy(update={"revision_state": RuntimeRevisionState.CANDIDATE})
        revision_service.persist_candidate_revision(candidate)
        validated = candidate.model_copy(
            update={
                "revision_state": RuntimeRevisionState.VALIDATED,
                "materialization_artifact_digest": artifact_digest,
            }
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


__all__ = [
    "ReferenceProductionLifecycleError",
    "ReferenceProductionLifecycleLauncher",
    "ReferenceProductionLifecycleResult",
    "ReferenceProductionLifecycleServices",
    "wire_reference_production_lifecycle_services",
]
