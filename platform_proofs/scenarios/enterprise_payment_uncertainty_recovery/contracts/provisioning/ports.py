"""Scenario provisioning port — plugin surface for interchangeable implementations."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.results import (
    CleanupPhaseOutcome,
    ProvisioningPhaseOutcome,
    ProvisioningPreparationHandle,
    ProvisioningSessionReference,
    StateAvailabilityOutcome,
)


class ScenarioProvisioningPort(Protocol):
    """Vendor-neutral provisioning capabilities: Prepare → Provision → State → Cleanup.

    Implementations materialize logical dataset truth into lab targets.
    The contract does not prescribe database, filesystem layout, or external APIs.
    """

    def prepare(self, context: ProvisioningContext) -> ProvisioningPhaseOutcome:
        """Validate dataset context and selected variant before materialization."""
        ...

    def provision(
        self,
        context: ProvisioningContext,
        preparation: ProvisioningPreparationHandle,
    ) -> ProvisioningPhaseOutcome:
        """Materialize shared and variant logical entities into configured targets."""
        ...

    def state_availability(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> StateAvailabilityOutcome:
        """Confirm provisioned state is available for scenario runtime entry."""
        ...

    def cleanup(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> CleanupPhaseOutcome:
        """Release or reset materialized lab state after evidence collection."""
        ...
