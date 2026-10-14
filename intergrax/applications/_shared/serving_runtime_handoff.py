# © Artur Czarnecki. All rights reserved.

"""Applications-owned serving runtime handoff (PROD-Q-R1-R7 ADR §4.5B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.activation import ActivationCommitResult
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
    ServingUnitAttestation,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.activation_domain_coordinator import (
    ActivationDomainCoordinator,
    ServingExecutionAdmissionRequest,
    ServingExecutionEligibilityEvidence,
)
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
    ServingExecutionAdmissionGrant,
    assert_activation_commit_matches_current_serving,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)
from intergrax.applications._shared.registry_projection import RegistryProjectionInputBundle
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolver,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.production_process_composition import (
        ProductionProcessComposition,
    )


class ServingRuntimeHandoffError(ValueError):
    """Serving runtime handoff establishment or validation failed."""


class ColocatedServingUnitRuntimeAccessor(Protocol):
    """VENV A1 wiring: map attested serving unit to revision-bound factory resolver."""

    def runtime_agent_factory_resolver_for_unit(
        self,
        *,
        serving_unit_ref: str,
        revision: RuntimeRevision,
        artifact_root: Path,
    ) -> RuntimeAgentFactoryResolver:
        """Return resolver for one warmed colocated unit; lifetime = process."""
        ...


class ColocatedServingUnitExecutionAdmission(Protocol):
    """Sanctioned execution accounting for one colocated serving unit (drain truth)."""

    def admit_execution(self, serving_unit_ref: str) -> None:
        """Record one authorized in-flight execution against the unit."""

    def release_execution(self, serving_unit_ref: str) -> None:
        """Release one in-flight execution slot."""


class ColocatedServingExecutionAdmissionAuthority(Protocol):
    """Opens physical execution admission on the deployment adapter after activation."""

    def authorize_serving_execution_admission(
        self, evidence: ServingExecutionAdmissionGrant
    ) -> None:
        """Authorize new execution work for one canonically eligible serving unit."""


class ServingUnitIdentity(BaseModel):
    """Attested serving unit identity carried into strict host consumption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    application_environment_id: str
    runtime_revision_id: str
    serving_unit_ref: str
    materialization_artifact_digest: str
    materialization_topology: MaterializationTopology


@dataclass(frozen=True, slots=True)
class ServingRuntimeHandoff:
    """Process-lifetime binding from attested unit to resolver + execution admission."""

    identity: ServingUnitIdentity
    runtime_agent_factory_resolver: RuntimeAgentFactoryResolver
    colocated_execution_admission: ColocatedServingUnitExecutionAdmission | None = None


def serving_unit_identity_from_attestation(
    attestation: ServingUnitAttestation,
    *,
    topology: MaterializationTopology,
) -> ServingUnitIdentity:
    return ServingUnitIdentity(
        application_id=attestation.application_id,
        application_environment_id=attestation.application_environment_id,
        runtime_revision_id=attestation.runtime_revision_id,
        serving_unit_ref=attestation.serving_unit_ref,
        materialization_artifact_digest=attestation.materialization_artifact_digest,
        materialization_topology=topology,
    )


def establish_serving_runtime_handoff(
    *,
    composition: ProductionProcessComposition,
    attestation: ServingUnitAttestation,
    deployment_instance: DeploymentInstanceRecord,
    projection_input: RegistryProjectionInputBundle,
    colocated_accessor: ColocatedServingUnitRuntimeAccessor | None,
    colocated_execution_admission: ColocatedServingUnitExecutionAdmission | None = None,
    topology: MaterializationTopology,
    artifact_root: Path | None = None,
) -> tuple[ServingRuntimeHandoff, ProductionProcessComposition]:
    """Bind attested unit identity to the authoritative projection resolver instance.

    Does not grant execution permission or mutate serving admission state.
    """
    from intergrax.applications._shared.production_process_composition import (
        ProductionProcessComposition,
    )

    if composition.serving_runtime_handoff is not None:
        raise ServingRuntimeHandoffError(
            "serving runtime handoff already established for this process composition"
        )
    if deployment_instance.serving_unit_ref != attestation.serving_unit_ref:
        raise ServingRuntimeHandoffError(
            "deployment instance serving_unit_ref mismatch with attestation"
        )
    if deployment_instance.runtime_revision_id != attestation.runtime_revision_id:
        raise ServingRuntimeHandoffError("deployment instance revision mismatch")

    projection_resolver = projection_input.factory_resolver
    if projection_resolver is None:
        raise ServingRuntimeHandoffError(
            "registry projection input bundle lacks factory_resolver"
        )

    if topology is MaterializationTopology.VENV_BUNDLE:
        if colocated_accessor is None:
            raise ServingRuntimeHandoffError(
                "VENV_BUNDLE handoff requires ColocatedServingUnitRuntimeAccessor"
            )
        if artifact_root is None:
            raise ServingRuntimeHandoffError(
                "VENV_BUNDLE handoff requires artifact_root for colocated accessor"
            )
        unit_resolver = colocated_accessor.runtime_agent_factory_resolver_for_unit(
            serving_unit_ref=attestation.serving_unit_ref,
            revision=projection_input.runtime_revision,
            artifact_root=artifact_root,
        )
        if unit_resolver is not projection_resolver:
            raise ServingRuntimeHandoffError(
                "colocated unit resolver is not the authoritative projection resolver instance"
            )
    elif colocated_accessor is not None:
        raise ServingRuntimeHandoffError(
            "colocated accessor is invalid for non-VENV_BUNDLE topology"
        )

    identity = serving_unit_identity_from_attestation(attestation, topology=topology)

    handoff = ServingRuntimeHandoff(
        identity=identity,
        runtime_agent_factory_resolver=projection_resolver,
        colocated_execution_admission=colocated_execution_admission,
    )
    updated = ProductionProcessComposition(
        agent_platform_runtime=composition.agent_platform_runtime,
        agent_capability_runtime=composition.agent_capability_runtime,
        provider_invocation_store=composition.provider_invocation_store,
        execution_continuation_state_store=composition.execution_continuation_state_store,
        serving_runtime_handoff=handoff,
    )
    return handoff, updated


def authorize_colocated_serving_execution_after_activation(
    *,
    activation_commit: ActivationCommitResult,
    handoff: ServingRuntimeHandoff,
    admission_authority: ColocatedServingExecutionAdmissionAuthority,
    activation_domain_coordinator: ActivationDomainCoordinator,
    serving_store: ApplicationEnvironmentServingStore,
    deployment_instance_store: DeploymentInstanceStore,
    revision_store: RuntimeRevisionStore,
) -> None:
    """Bridge authoritative activation commit evidence to physical execution admission."""
    identity = handoff.identity
    serving = activation_commit.serving_record
    activated = activation_commit.activated_revision
    candidate = activation_commit.candidate_instance

    if activated.revision_state is not RuntimeRevisionState.ACTIVE:
        raise ServingRuntimeHandoffError(
            "activated revision is not active for execution admission"
        )
    if candidate.instance_state is not DeploymentInstanceState.SERVING:
        raise ServingRuntimeHandoffError(
            "activation commit deployment instance is not serving"
        )

    if serving.traffic_serving_revision_id != identity.runtime_revision_id:
        raise ServingRuntimeHandoffError(
            "activation serving pointer does not match handoff runtime revision"
        )
    if serving.application_id != identity.application_id:
        raise ServingRuntimeHandoffError(
            "activation serving record application mismatch with handoff identity"
        )
    if serving.application_environment_id != identity.application_environment_id:
        raise ServingRuntimeHandoffError(
            "activation serving record environment mismatch with handoff identity"
        )
    if activated.runtime_revision_id != identity.runtime_revision_id:
        raise ServingRuntimeHandoffError(
            "activated revision does not match handoff identity"
        )
    if activated.materialization_artifact_digest != identity.materialization_artifact_digest:
        raise ServingRuntimeHandoffError(
            "activated revision digest does not match handoff identity"
        )
    if candidate.serving_unit_ref != identity.serving_unit_ref:
        raise ServingRuntimeHandoffError(
            "candidate deployment instance serving unit mismatch with handoff"
        )
    if candidate.runtime_revision_id != identity.runtime_revision_id:
        raise ServingRuntimeHandoffError(
            "candidate deployment instance revision mismatch with handoff"
        )

    current_serving = serving_store.get_serving_record(
        identity.application_id,
        identity.application_environment_id,
    )
    if current_serving is None:
        raise ServingRuntimeHandoffError("canonical serving record is missing")
    try:
        assert_activation_commit_matches_current_serving(
            activation_commit_serving=serving,
            current_serving=current_serving,
        )
    except ServingExecutionAdmissionError as exc:
        raise ServingRuntimeHandoffError(str(exc)) from exc

    request = ServingExecutionAdmissionRequest(
        application_id=identity.application_id,
        application_environment_id=identity.application_environment_id,
        runtime_revision_id=identity.runtime_revision_id,
        serving_unit_ref=identity.serving_unit_ref,
        materialization_artifact_digest=identity.materialization_artifact_digest,
    )

    def _open_admission(evidence: ServingExecutionEligibilityEvidence) -> None:
        if evidence.serving_pointer_revision != serving.serving_pointer_revision:
            raise ServingRuntimeHandoffError(
                "serving pointer revision mismatch with eligibility evidence"
            )
        admission_authority.authorize_serving_execution_admission(evidence)

    try:
        activation_domain_coordinator.coordinate_unit_serving_admission_open(
            request,
            open_unit_admission=_open_admission,
        )
    except ServingExecutionAdmissionError as exc:
        raise ServingRuntimeHandoffError(str(exc)) from exc


def validate_serving_runtime_handoff_against_projection_evidence(
    handoff: ServingRuntimeHandoff,
    *,
    runtime_revision_id: str,
    materialization_artifact_digest: str | None,
) -> None:
    """Fail closed when strict host projection evidence disagrees with handoff identity."""
    if handoff.identity.runtime_revision_id != runtime_revision_id:
        raise ServingRuntimeHandoffError(
            "serving runtime handoff revision disagrees with registry projection evidence"
        )
    if (
        materialization_artifact_digest is not None
        and handoff.identity.materialization_artifact_digest
        != materialization_artifact_digest
    ):
        raise ServingRuntimeHandoffError(
            "serving runtime handoff digest disagrees with registry projection evidence"
        )


__all__ = [
    "ColocatedServingExecutionAdmissionAuthority",
    "ColocatedServingUnitExecutionAdmission",
    "ColocatedServingUnitRuntimeAccessor",
    "ServingRuntimeHandoff",
    "ServingRuntimeHandoffError",
    "ServingUnitIdentity",
    "authorize_colocated_serving_execution_after_activation",
    "establish_serving_runtime_handoff",
    "serving_unit_identity_from_attestation",
    "validate_serving_runtime_handoff_against_projection_evidence",
]
