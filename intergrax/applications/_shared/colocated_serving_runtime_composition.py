# © Artur Czarnecki. All rights reserved.

"""Sanctioned production composition bridge: activation commit → handoff → execution admission."""

from __future__ import annotations

from pathlib import Path

from intergrax.agent_distribution.activation import ActivationCommitResult
from intergrax.agent_distribution.activation_domain_coordinator import (
    ActivationDomainCoordinator,
    ActivationDomainCoordinatorError,
    build_activation_domain_coordinator_for_stores,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    RuntimeDeploymentAdapter,
    ServingUnitAttestation,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    StoreBackedColocatedServingUnitExecutionAdmission,
)
from intergrax.applications._shared.colocated_unit_physical_execution_gate import (
    as_colocated_unit_physical_gate,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.serving_runtime_handoff import (
    ColocatedServingUnitRuntimeAccessor,
    ServingRuntimeHandoff,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)


class ColocatedServingAdmissionTopologyError(ValueError):
    """Unsupported or inconsistent colocated serving admission wiring."""


class ColocatedServingActivationCommitResolutionError(ValueError):
    """Canonical activation commit evidence could not be resolved from durable stores."""


def _physical_unit_already_serving_for_attestation(
    physical_adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    attestation: ServingUnitAttestation,
) -> bool:
    """True when a live unit is already execution-eligible for the attested identity."""
    unit = physical_adapter._units.get(attestation.serving_unit_ref)
    if unit is None:
        return False
    if not unit.serving_admission_authorized or unit.admission.admission_closed():
        return False
    if unit.lifecycle_state is not DeploymentInstanceState.SERVING:
        return False
    return (
        unit.application_id == attestation.application_id
        and unit.application_environment_id == attestation.application_environment_id
        and unit.runtime_revision_id == attestation.runtime_revision_id
        and unit.serving_unit_ref == attestation.serving_unit_ref
        and unit.materialization_artifact_digest == attestation.materialization_artifact_digest
    )


def resolve_traffic_serving_activation_commit(
    *,
    serving_store: ApplicationEnvironmentServingStore,
    revision_store: RuntimeRevisionStore,
    deployment_instance_store: DeploymentInstanceStore,
    application_id: str,
    application_environment_id: str,
) -> ActivationCommitResult:
    """Reconstruct activation commit evidence for the current traffic-serving revision."""
    serving = serving_store.get_serving_record(application_id, application_environment_id)
    if serving is None:
        raise ColocatedServingActivationCommitResolutionError(
            "missing durable serving pointer for application environment",
        )
    revision_id = serving.traffic_serving_revision_id
    if revision_id is None:
        raise ColocatedServingActivationCommitResolutionError(
            "serving pointer lacks traffic_serving_revision_id",
        )
    revision = revision_store.get_revision(revision_id)
    if revision is None:
        raise ColocatedServingActivationCommitResolutionError(
            f"missing runtime revision {revision_id!r} for serving pointer",
        )
    if revision.revision_state is not RuntimeRevisionState.ACTIVE:
        raise ColocatedServingActivationCommitResolutionError(
            "traffic-serving revision is not active",
        )
    if (
        revision.application_id != application_id
        or revision.application_environment_id != application_environment_id
    ):
        raise ColocatedServingActivationCommitResolutionError(
            "traffic-serving revision scope does not match application environment",
        )
    instance = deployment_instance_store.get_instance(
        application_id,
        application_environment_id,
        revision_id,
    )
    if instance is None:
        raise ColocatedServingActivationCommitResolutionError(
            "missing deployment instance for traffic-serving revision",
        )
    if instance.instance_state is not DeploymentInstanceState.SERVING:
        raise ColocatedServingActivationCommitResolutionError(
            "deployment instance for traffic-serving revision is not serving",
        )
    if instance.serving_unit_ref is None:
        raise ColocatedServingActivationCommitResolutionError(
            "deployment instance lacks serving_unit_ref",
        )
    return ActivationCommitResult(
        serving_record=serving,
        activated_revision=revision,
        candidate_instance=instance,
    )


def validate_colocated_serving_admission_topology(
    *,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
) -> ActivationDomainCoordinator:
    """Fail closed when store bundle cannot share one activation domain coordinator."""
    try:
        return build_activation_domain_coordinator_for_stores(
            serving_store=serving_store,
            deployment_instance_store=deployment_instance_store,
            revision_store=revision_store,
        )
    except ActivationDomainCoordinatorError as exc:
        raise ColocatedServingAdmissionTopologyError(str(exc)) from exc


def bind_colocated_serving_runtime_after_activation(
    *,
    composition: ProductionProcessComposition,
    activation_commit: ActivationCommitResult,
    attestation: ServingUnitAttestation,
    deployment_instance: DeploymentInstanceRecord,
    projection_input: RegistryProjectionInputBundle,
    colocated_accessor: ColocatedServingUnitRuntimeAccessor,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
    topology: MaterializationTopology,
    artifact_root: Path,
    physical_adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
) -> ProductionProcessComposition:
    """Establish typed handoff, then open physical admission using activation evidence only."""
    activation_domain_coordinator = validate_colocated_serving_admission_topology(
        serving_store=serving_store,
        deployment_instance_store=deployment_instance_store,
        revision_store=revision_store,
    )
    handoff, updated = establish_serving_runtime_handoff(
        composition=composition,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=projection_input,
        colocated_accessor=colocated_accessor,
        colocated_execution_admission=None,
        topology=topology,
        artifact_root=artifact_root,
    )
    physical_gate = as_colocated_unit_physical_gate(physical_adapter)
    unit_admission_open = physical_adapter.coordinator_verified_unit_serving_admission_open_port(
        attestation.serving_unit_ref
    )
    store_backed_admission = StoreBackedColocatedServingUnitExecutionAdmission(
        physical_gate=physical_gate,
        activation_domain_coordinator=activation_domain_coordinator,
        identity=handoff.identity,
    )
    handoff = handoff.__class__(
        identity=handoff.identity,
        runtime_agent_factory_resolver=handoff.runtime_agent_factory_resolver,
        colocated_execution_admission=store_backed_admission,
    )
    updated = updated.__class__(
        agent_platform_runtime=updated.agent_platform_runtime,
        agent_capability_runtime=updated.agent_capability_runtime,
        provider_invocation_store=updated.provider_invocation_store,
        execution_continuation_state_store=updated.execution_continuation_state_store,
        serving_runtime_handoff=handoff,
    )
    if not _physical_unit_already_serving_for_attestation(physical_adapter, attestation):
        authorize_colocated_serving_execution_after_activation(
            activation_commit=activation_commit,
            handoff=handoff,
            unit_admission_open=unit_admission_open,
            activation_domain_coordinator=activation_domain_coordinator,
            serving_store=serving_store,
            deployment_instance_store=deployment_instance_store,
            revision_store=revision_store,
        )
    return updated


def begin_colocated_serving_execution_drain_on_strict_session_close(
    *,
    deployment_adapter: RuntimeDeploymentAdapter,
    handoff: ServingRuntimeHandoff,
    revision_store: RuntimeRevisionStore,
) -> None:
    """Revoke new physical execution admission for the bound handoff unit (session close owner)."""
    if handoff.colocated_execution_admission is None:
        return
    if not isinstance(deployment_adapter, VenvBundleColocatedRuntimeDeploymentAdapter):
        raise ColocatedServingAdmissionTopologyError(
            "strict session close requires VenvBundleColocatedRuntimeDeploymentAdapter",
        )
    identity = handoff.identity
    revision = revision_store.get_revision(identity.runtime_revision_id)
    if revision is None:
        raise ColocatedServingAdmissionTopologyError(
            f"missing runtime revision {identity.runtime_revision_id!r} for session close drain",
        )
    deployment_adapter.begin_drain(
        revision,
        serving_unit_ref=identity.serving_unit_ref,
    )


__all__ = [
    "ColocatedServingActivationCommitResolutionError",
    "ColocatedServingAdmissionTopologyError",
    "begin_colocated_serving_execution_drain_on_strict_session_close",
    "bind_colocated_serving_runtime_after_activation",
    "resolve_traffic_serving_activation_commit",
    "validate_colocated_serving_admission_topology",
]
