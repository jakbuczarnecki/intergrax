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
    ServingUnitAttestation,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
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
    ColocatedServingExecutionAdmissionAuthority,
    ColocatedServingUnitRuntimeAccessor,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)


class ColocatedServingAdmissionTopologyError(ValueError):
    """Unsupported or inconsistent colocated serving admission wiring."""


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
    admission_authority: ColocatedServingExecutionAdmissionAuthority,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
    topology: MaterializationTopology,
    artifact_root: Path,
    physical_adapter: VenvBundleColocatedRuntimeDeploymentAdapter | None = None,
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
    if physical_adapter is None and isinstance(
        admission_authority, VenvBundleColocatedRuntimeDeploymentAdapter
    ):
        physical_adapter = admission_authority
    if physical_adapter is None:
        raise ColocatedServingAdmissionTopologyError(
            "physical deployment adapter is required for colocated execution gate wiring"
        )
    physical_gate = as_colocated_unit_physical_gate(physical_adapter)
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
    authorize_colocated_serving_execution_after_activation(
        activation_commit=activation_commit,
        handoff=handoff,
        admission_authority=admission_authority,
        activation_domain_coordinator=activation_domain_coordinator,
        serving_store=serving_store,
        deployment_instance_store=deployment_instance_store,
        revision_store=revision_store,
    )
    return updated


__all__ = [
    "ColocatedServingAdmissionTopologyError",
    "ColocatedServingExecutionAdmissionAuthority",
    "bind_colocated_serving_runtime_after_activation",
    "validate_colocated_serving_admission_topology",
]
