"""Typed provisioning phase outcomes and session references for downstream execution."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.failures import (
    ProvisioningFailure,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
    ProvisioningPhase,
)


@dataclass(frozen=True, slots=True)
class ProvisioningPreparationHandle:
    """Harness-visible result of Prepare — required input to Provision."""

    qualification_id: str
    variant_id: str
    logical_fingerprint: str


@dataclass(frozen=True, slots=True)
class ProvisioningSessionReference:
    """Opaque handles for proof harness / runtime wiring after materialization."""

    session_id: str
    variant_id: str
    logical_fingerprint: str
    execution_handles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProvisioningPhaseOutcome:
    """Result of Prepare or Provision."""

    phase: ProvisioningPhase
    status: ProvisioningOutcomeStatus
    preparation: ProvisioningPreparationHandle | None = None
    session: ProvisioningSessionReference | None = None
    failure: ProvisioningFailure | None = None


@dataclass(frozen=True, slots=True)
class StateAvailabilityOutcome:
    """Result of State Availability — confirms provisioned state is ready for runtime entry."""

    phase: ProvisioningPhase
    status: ProvisioningOutcomeStatus
    session: ProvisioningSessionReference
    state_ready: bool
    failure: ProvisioningFailure | None = None


@dataclass(frozen=True, slots=True)
class CleanupPhaseOutcome:
    """Result of Cleanup — tear-down or reset of lab materialization."""

    phase: ProvisioningPhase
    status: ProvisioningOutcomeStatus
    failure: ProvisioningFailure | None = None
