"""PostgreSQL implementation of ScenarioProvisioningPort."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.infrastructure.contract import (
    POSTGRES_ENV_FILE,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.connection import (
    connect,
    load_connection_settings,
    postgres_lab_available,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.dataset_loader import (
    load_scenario_package,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materialization import (
    MaterializedScenarioState,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.materializer import (
    cleanup_state,
    materialize_package,
    state_records_present,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.postgresql.schema_bootstrap import (
    ensure_lab_schema,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    InvalidDatasetError,
    MissingScenarioVariantError,
)


@dataclass
class PostgreSqlScenarioProvisioner:
    """Materialize dataset/ into the ERL-QUAL-004 lab PostgreSQL instance."""

    env_file: Path = POSTGRES_ENV_FILE
    _sessions: dict[str, MaterializedScenarioState] = field(default_factory=dict)

    def prepare(self, context: ProvisioningContext) -> ProvisioningPhaseOutcome:
        try:
            resolution = load_scenario_package(
                dataset_package_root=context.execution.dataset_package_root,
                qualification_id=context.identity.qualification_id,
                scenario_slug=context.identity.scenario_slug,
                variant_id=context.variant.variant_id,
            ).resolution
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

        if not postgres_lab_available(self.env_file):
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.PREPARE,
                    message="PostgreSQL lab infrastructure is not available",
                ),
            )

        settings = load_connection_settings(self.env_file)
        if settings is None:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.PREPARE,
                    message="PostgreSQL lab configuration is invalid or missing",
                ),
            )

        try:
            with connect(settings) as conn:
                ensure_lab_schema(conn)
        except Exception as exc:  # noqa: BLE001 — surfaced as typed provisioning failure
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PREPARE,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.PREPARE,
                    message=f"database unavailable during prepare: {exc}",
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

        settings = load_connection_settings(self.env_file)
        if settings is None:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PROVISION,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.PROVISION,
                    message="PostgreSQL lab configuration is invalid or missing",
                ),
            )

        try:
            package = load_scenario_package(
                dataset_package_root=context.execution.dataset_package_root,
                qualification_id=context.identity.qualification_id,
                scenario_slug=context.identity.scenario_slug,
                variant_id=context.variant.variant_id,
            )
        except (InvalidDatasetError, MissingScenarioVariantError) as exc:
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PROVISION,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INVALID_DATASET,
                    phase=ProvisioningPhase.PROVISION,
                    message=str(exc),
                ),
            )

        try:
            with connect(settings) as conn:
                ensure_lab_schema(conn)
                materialized = materialize_package(conn, package)
        except Exception as exc:  # noqa: BLE001
            return ProvisioningPhaseOutcome(
                phase=ProvisioningPhase.PROVISION,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INCOMPLETE_PROVISIONING,
                    phase=ProvisioningPhase.PROVISION,
                    message=f"provisioning failed: {exc}",
                ),
            )

        session_id = f"pg-{context.execution.run_id}-{uuid4().hex[:12]}"
        self._sessions[session_id] = materialized
        session = ProvisioningSessionReference(
            session_id=session_id,
            variant_id=preparation.variant_id,
            logical_fingerprint=preparation.logical_fingerprint,
            execution_handles=materialized.execution_handles(),
        )
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
        materialized = self._sessions.get(session.session_id)
        if materialized is None:
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

        settings = load_connection_settings(self.env_file)
        if settings is None:
            return StateAvailabilityOutcome(
                phase=ProvisioningPhase.STATE_AVAILABILITY,
                status=ProvisioningOutcomeStatus.FAILED,
                session=session,
                state_ready=False,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.PROVISIONING_UNAVAILABLE,
                    phase=ProvisioningPhase.STATE_AVAILABILITY,
                    message="PostgreSQL lab configuration is invalid or missing",
                ),
            )

        try:
            with connect(settings) as conn:
                ready = state_records_present(conn, materialized)
        except Exception as exc:  # noqa: BLE001
            return StateAvailabilityOutcome(
                phase=ProvisioningPhase.STATE_AVAILABILITY,
                status=ProvisioningOutcomeStatus.FAILED,
                session=session,
                state_ready=False,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INCOMPLETE_PROVISIONING,
                    phase=ProvisioningPhase.STATE_AVAILABILITY,
                    message=f"state verification failed: {exc}",
                ),
            )

        if not ready:
            return StateAvailabilityOutcome(
                phase=ProvisioningPhase.STATE_AVAILABILITY,
                status=ProvisioningOutcomeStatus.FAILED,
                session=session,
                state_ready=False,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.INCOMPLETE_PROVISIONING,
                    phase=ProvisioningPhase.STATE_AVAILABILITY,
                    message="required scenario rows are missing",
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
        materialized = self._sessions.pop(session.session_id, None)
        if materialized is None:
            return CleanupPhaseOutcome(
                phase=ProvisioningPhase.CLEANUP,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.CLEANUP_FAILURE,
                    phase=ProvisioningPhase.CLEANUP,
                    message="session already released or never provisioned",
                ),
            )

        settings = load_connection_settings(self.env_file)
        if settings is None:
            return CleanupPhaseOutcome(
                phase=ProvisioningPhase.CLEANUP,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.CLEANUP_FAILURE,
                    phase=ProvisioningPhase.CLEANUP,
                    message="PostgreSQL lab configuration is invalid or missing",
                ),
            )

        try:
            with connect(settings) as conn:
                cleanup_state(conn, materialized)
        except Exception as exc:  # noqa: BLE001
            self._sessions[session.session_id] = materialized
            return CleanupPhaseOutcome(
                phase=ProvisioningPhase.CLEANUP,
                status=ProvisioningOutcomeStatus.FAILED,
                failure=ProvisioningFailure(
                    code=ProvisioningFailureCode.CLEANUP_FAILURE,
                    phase=ProvisioningPhase.CLEANUP,
                    message=f"cleanup failed: {exc}",
                ),
            )

        return CleanupPhaseOutcome(
            phase=ProvisioningPhase.CLEANUP,
            status=ProvisioningOutcomeStatus.SUCCEEDED,
        )
