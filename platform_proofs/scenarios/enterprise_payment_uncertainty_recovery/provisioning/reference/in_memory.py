"""In-memory reference provisioner — materializes logical handles without lab storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.context import (
    ProvisioningContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.failures import (
    ProvisioningFailure,
    ProvisioningFailureCode,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
    ProvisioningPhase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.results import (
    CleanupPhaseOutcome,
    ProvisioningPhaseOutcome,
    ProvisioningPreparationHandle,
    ProvisioningSessionReference,
    StateAvailabilityOutcome,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    InvalidDatasetError,
    MissingScenarioVariantError,
    resolve_variant_from_manifest,
)


@dataclass
class InMemoryReferenceProvisioner:
    """Scenario-local reference implementation — replaceable, not a registry singleton."""

    _active_sessions: dict[str, ProvisioningSessionReference] = field(default_factory=dict)
    unavailable: bool = False

    def prepare(self, context: ProvisioningContext) -> ProvisioningPhaseOutcome:
        if self.unavailable:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.PREPARE,
                    message="reference provisioner marked unavailable",
                ),
            )
        try:
            resolution = resolve_variant_from_manifest(
                dataset_package_root=context.execution.dataset_package_root,
                qualification_id=context.identity.qualification_id,
                scenario_slug=context.identity.scenario_slug,
                variant_id=context.variant.variant_id,
            )
        except MissingScenarioVariantError as exc:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.MISSING_SCENARIO_VARIANT,
                    phase=ProvisioningPhase.PREPARE,
                    message=str(exc),
                ),
            )
        except InvalidDatasetError as exc:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INVALID_DATASET,
                    phase=ProvisioningPhase.PREPARE,
                    message=str(exc),
                ),
            )

        handle = ProvisioningPreparationHandle(
            qualification_id=resolution.qualification_id,
            variant_id=resolution.variant.variant_id,
            logical_fingerprint=resolution.variant.logical_fingerprint,
        )
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PREPARE,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            preparation=handle,
        )

    def provision(
        self,
        context: ProvisioningContext,
        preparation: ProvisioningPreparationHandle,
    ) -> ProvisioningPhaseOutcome:
        if preparation.variant_id != context.variant.variant_id:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PROVISION,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INCONSISTENT_ENVIRONMENT,
                    phase=ProvisioningPhase.PROVISION,
                    message="preparation handle variant does not match context",
                ),
            )

        session_id = f"ref-{context.execution.run_id}-{uuid4().hex[:12]}"
        session = ProvisioningSessionReference(
            session_id=session_id,
            variant_id=preparation.variant_id,
            logical_fingerprint=preparation.logical_fingerprint,
            execution_handles=(
                f"logical-variant:{preparation.variant_id}",
                f"run:{context.execution.run_id}",
            ),
        )
        self._active_sessions[session_id] = session
        return ProvisioningPhaseOutcome(
            phase=ProvisioningPhase.PROVISION,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
        )

    def state_availability(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> StateAvailabilityOutcome:
        stored = self._active_sessions.get(session.session_id)
        if stored is None:
            return StateAvailabilityOutcome(
                phase=ProvisioningPhase.STATE_AVAILABILITY,
                status=ProvisioningOutcomeStatus.FAILED,
                session=session,
                state_ready=False,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INCOMPLETE_PROVISIONING,
                    phase=ProvisioningPhase.STATE_AVAILABILITY,
                    message="session not found after provision",
                ),
            )
        return StateAvailabilityOutcome(
            phase=ProvisioningPhase.STATE_AVAILABILITY,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
            session=session,
            state_ready=True,
        )

    def cleanup(
        self,
        context: ProvisioningContext,
        session: ProvisioningSessionReference,
    ) -> CleanupPhaseOutcome:
        if session.session_id not in self._active_sessions:
            return CleanupPhaseOutcome(
                phase=ProvisioningPhase.CLEANUP,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.CLEANUP_FAILURE,
                    phase=ProvisioningPhase.CLEANUP,
                    message="session already released or never provisioned",
                ),
            )
        del self._active_sessions[session.session_id]
        return CleanupPhaseOutcome(
            phase=ProvisioningPhase.CLEANUP,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
        )
