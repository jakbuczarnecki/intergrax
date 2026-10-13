# © Artur Czarnecki. All rights reserved.

"""Store-backed colocated serving unit execution admission (PROD-Q-R1-R7-D1-R3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.agent_distribution.serving_execution_admission import (
    verify_current_serving_execution_admission_eligibility,
)
from intergrax.agent_distribution.stores import (
    ApplicationEnvironmentServingStore,
    DeploymentInstanceStore,
    RuntimeRevisionStore,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoffError,
    ServingUnitIdentity,
)


class ColocatedUnitPhysicalExecutionGate(Protocol):
    """Narrow delegate for physical admit/release without identifier-only authority."""

    def admit_execution(self, serving_unit_ref: str) -> None: ...

    def release_execution(self, serving_unit_ref: str) -> None: ...


@dataclass(frozen=True, slots=True)
class StoreBackedColocatedServingUnitExecutionAdmission:
    """Per-execution admission gated on current canonical serving eligibility."""

    physical_gate: ColocatedUnitPhysicalExecutionGate
    serving_store: ApplicationEnvironmentServingStore
    deployment_instance_store: DeploymentInstanceStore
    revision_store: RuntimeRevisionStore
    identity: ServingUnitIdentity

    def admit_execution(self, serving_unit_ref: str) -> None:
        if serving_unit_ref != self.identity.serving_unit_ref:
            raise ServingRuntimeHandoffError(
                "execution admission serving unit ref disagrees with handoff identity"
            )
        verify_current_serving_execution_admission_eligibility(
            serving_store=self.serving_store,
            deployment_instance_store=self.deployment_instance_store,
            revision_store=self.revision_store,
            application_id=self.identity.application_id,
            application_environment_id=self.identity.application_environment_id,
            runtime_revision_id=self.identity.runtime_revision_id,
            serving_unit_ref=self.identity.serving_unit_ref,
            materialization_artifact_digest=self.identity.materialization_artifact_digest,
        )
        self.physical_gate.admit_execution(serving_unit_ref)

    def release_execution(self, serving_unit_ref: str) -> None:
        if serving_unit_ref != self.identity.serving_unit_ref:
            raise ServingRuntimeHandoffError(
                "execution release serving unit ref disagrees with handoff identity"
            )
        self.physical_gate.release_execution(serving_unit_ref)


__all__ = ["StoreBackedColocatedServingUnitExecutionAdmission"]
