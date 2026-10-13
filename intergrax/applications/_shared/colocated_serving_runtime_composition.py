# © Artur Czarnecki. All rights reserved.

"""Sanctioned production composition bridge: activation commit → handoff → execution admission."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from intergrax.agent_distribution.activation import ActivationCommitResult
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
from intergrax.applications._shared.colocated_serving_unit_execution_admission import (
    ColocatedUnitPhysicalExecutionGate,
    StoreBackedColocatedServingUnitExecutionAdmission,
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
) -> ProductionProcessComposition:
    """Establish typed handoff, then open physical admission using activation evidence only."""
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
    store_backed_admission = StoreBackedColocatedServingUnitExecutionAdmission(
        physical_gate=cast(ColocatedUnitPhysicalExecutionGate, colocated_accessor),
        serving_store=serving_store,
        deployment_instance_store=deployment_instance_store,
        revision_store=revision_store,
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
        serving_store=serving_store,
        deployment_instance_store=deployment_instance_store,
        revision_store=revision_store,
    )
    return updated


__all__ = [
    "ColocatedServingExecutionAdmissionAuthority",
    "bind_colocated_serving_runtime_after_activation",
]
