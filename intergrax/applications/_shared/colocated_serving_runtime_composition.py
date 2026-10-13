# © Artur Czarnecki. All rights reserved.

"""Sanctioned production composition bridge: activation commit → handoff → execution admission."""

from __future__ import annotations

from pathlib import Path

from intergrax.agent_distribution.activation import ActivationCommitResult
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    ServingUnitAttestation,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.serving_runtime_handoff import (
    ColocatedServingExecutionAdmissionAuthority,
    ColocatedServingUnitExecutionAdmission,
    ColocatedServingUnitRuntimeAccessor,
    authorize_colocated_serving_execution_after_activation,
    establish_serving_runtime_handoff,
)


def bind_colocated_serving_runtime_after_activation(
    *,
    composition: ProductionProcessComposition,
    activation_commit: ActivationCommitResult,
    attestation: ServingUnitAttestation,
    deployment_instance: DeploymentInstanceRecord,
    projection_input: RegistryProjectionInputBundle,
    colocated_accessor: ColocatedServingUnitRuntimeAccessor,
    admission_authority: ColocatedServingExecutionAdmissionAuthority,
    colocated_execution_admission: ColocatedServingUnitExecutionAdmission,
    topology: MaterializationTopology,
    artifact_root: Path,
) -> ProductionProcessComposition:
    """Establish typed handoff, then open physical admission using activation evidence only."""
    handoff, updated = establish_serving_runtime_handoff(
        composition=composition,
        attestation=attestation,
        deployment_instance=deployment_instance,
        projection_input=projection_input,
        colocated_accessor=colocated_accessor,
        colocated_execution_admission=colocated_execution_admission,
        topology=topology,
        artifact_root=artifact_root,
    )
    authorize_colocated_serving_execution_after_activation(
        activation_commit=activation_commit,
        handoff=handoff,
        admission_authority=admission_authority,
    )
    return updated


__all__ = [
    "ColocatedServingExecutionAdmissionAuthority",
    "bind_colocated_serving_runtime_after_activation",
]
