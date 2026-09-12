"""Provisioning lifecycle coordinator — enforces phase order without vendor coupling."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.ports import (
    ScenarioProvisioningPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.results import (
    CleanupPhaseOutcome,
    ProvisioningPhaseOutcome,
    ProvisioningSessionReference,
    StateAvailabilityOutcome,
)


@dataclass(frozen=True, slots=True)
class ProvisioningLifecycleRunResult:
    """Aggregate outcome of a full provisioning lifecycle invocation."""

    status: ProvisioningOutcomeStatus
    prepare: ProvisioningPhaseOutcome | None
    provision: ProvisioningPhaseOutcome | None
    state_availability: StateAvailabilityOutcome | None
    cleanup: CleanupPhaseOutcome | None
    session: ProvisioningSessionReference | None


class ProvisioningLifecycleCoordinator:
    """Proof-harness-facing orchestrator: runs provisioning phases in architecture order."""

    def run(
        self,
        port: ScenarioProvisioningPort,
        context: ProvisioningContext,
        *,
        invoke_cleanup: bool = True,
    ) -> ProvisioningLifecycleRunResult:
        prepare_outcome = port.prepare(context)
        if prepare_outcome.status is ProvisioningOutcomeStatus.FAILED:
            cleanup_outcome = self._maybe_cleanup_after_partial(
                port, context, invoke_cleanup=invoke_cleanup, session=None
            )
            return ProvisioningLifecycleRunResult(
                status=ProvisioningOutcomeStatus.FAILED,
                prepare=prepare_outcome,
                provision=None,
                state_availability=None,
                cleanup=cleanup_outcome,
                session=None,
            )

        preparation = prepare_outcome.preparation
        if preparation is None:
            return ProvisioningLifecycleRunResult(
                status=ProvisioningOutcomeStatus.FAILED,
                prepare=prepare_outcome,
                provision=None,
                state_availability=None,
                cleanup=None,
                session=None,
            )

        provision_outcome = port.provision(context, preparation)
        if provision_outcome.status is ProvisioningOutcomeStatus.FAILED:
            session = provision_outcome.session
            cleanup_outcome = self._maybe_cleanup_after_partial(
                port, context, invoke_cleanup=invoke_cleanup, session=session
            )
            return ProvisioningLifecycleRunResult(
                status=ProvisioningOutcomeStatus.FAILED,
                prepare=prepare_outcome,
                provision=provision_outcome,
                state_availability=None,
                cleanup=cleanup_outcome,
                session=session,
            )

        session = provision_outcome.session
        if session is None:
            return ProvisioningLifecycleRunResult(
                status=ProvisioningOutcomeStatus.FAILED,
                prepare=prepare_outcome,
                provision=provision_outcome,
                state_availability=None,
                cleanup=None,
                session=None,
            )

        availability_outcome = port.state_availability(context, session)
        if availability_outcome.status is ProvisioningOutcomeStatus.FAILED:
            cleanup_outcome = self._maybe_cleanup_after_partial(
                port, context, invoke_cleanup=invoke_cleanup, session=session
            )
            return ProvisioningLifecycleRunResult(
                status=ProvisioningOutcomeStatus.FAILED,
                prepare=prepare_outcome,
                provision=provision_outcome,
                state_availability=availability_outcome,
                cleanup=cleanup_outcome,
                session=session,
            )

        cleanup_outcome: CleanupPhaseOutcome | None = None
        if invoke_cleanup:
            cleanup_outcome = port.cleanup(context, session)
            if cleanup_outcome.status is ProvisioningOutcomeStatus.FAILED:
                return ProvisioningLifecycleRunResult(
                    status=ProvisioningOutcomeStatus.FAILED,
                    prepare=prepare_outcome,
                    provision=provision_outcome,
                    state_availability=availability_outcome,
                    cleanup=cleanup_outcome,
                    session=session,
                )

        return ProvisioningLifecycleRunResult(
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            prepare=prepare_outcome,
            provision=provision_outcome,
            state_availability=availability_outcome,
            cleanup=cleanup_outcome,
            session=session,
        )

    @staticmethod
    def _maybe_cleanup_after_partial(
        port: ScenarioProvisioningPort,
        context: ProvisioningContext,
        *,
        invoke_cleanup: bool,
        session: ProvisioningSessionReference | None,
    ) -> CleanupPhaseOutcome | None:
        if not invoke_cleanup or session is None:
            return None
        return port.cleanup(context, session)
