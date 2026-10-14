# © Artur Czarnecki. All rights reserved.

"""Store-backed colocated serving unit execution admission (PROD-Q-R1-R7-D1-R6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.agent_distribution.activation_domain_coordinator import (
    ActivationDomainCoordinator,
    ServingExecutionAdmissionRequest,
)
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoffError,
    ServingUnitIdentity,
)


class ColocatedUnitPhysicalExecutionGate(Protocol):
    """Narrow delegate for physical admit/release without identifier-only authority."""

    def admit_execution(self, serving_unit_ref: str) -> None: ...

    def release_execution(self, serving_unit_ref: str) -> None: ...


def serving_execution_admission_request_from_identity(
    identity: ServingUnitIdentity,
) -> ServingExecutionAdmissionRequest:
    return ServingExecutionAdmissionRequest(
        application_id=identity.application_id,
        application_environment_id=identity.application_environment_id,
        runtime_revision_id=identity.runtime_revision_id,
        serving_unit_ref=identity.serving_unit_ref,
        materialization_artifact_digest=identity.materialization_artifact_digest,
    )


@dataclass(frozen=True, slots=True)
class StoreBackedColocatedServingUnitExecutionAdmission:
    """Per-execution admission gated on current canonical serving eligibility."""

    physical_gate: ColocatedUnitPhysicalExecutionGate
    activation_domain_coordinator: ActivationDomainCoordinator
    identity: ServingUnitIdentity

    def admit_execution(self, serving_unit_ref: str) -> None:
        if serving_unit_ref != self.identity.serving_unit_ref:
            raise ServingRuntimeHandoffError(
                "execution admission serving unit ref disagrees with handoff identity"
            )
        request = serving_execution_admission_request_from_identity(self.identity)
        try:
            self.activation_domain_coordinator.coordinate_physical_execution_reservation(
                request,
                physical_reserve=lambda: self.physical_gate.admit_execution(serving_unit_ref),
                physical_release=lambda: self.physical_gate.release_execution(serving_unit_ref),
            )
        except ServingExecutionAdmissionError as exc:
            raise ServingRuntimeHandoffError(str(exc)) from exc

    def release_execution(self, serving_unit_ref: str) -> None:
        if serving_unit_ref != self.identity.serving_unit_ref:
            raise ServingRuntimeHandoffError(
                "execution release serving unit ref disagrees with handoff identity"
            )
        self.physical_gate.release_execution(serving_unit_ref)


__all__ = [
    "ColocatedUnitPhysicalExecutionGate",
    "StoreBackedColocatedServingUnitExecutionAdmission",
    "serving_execution_admission_request_from_identity",
]
