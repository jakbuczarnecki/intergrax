# © Artur Czarnecki. All rights reserved.

"""Post-restart serving reconciliation for production process lifecycle (PROD-Q-R1-R7-D3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from intergrax.agent_distribution.deployment import ServingUnitAttestation
from intergrax.agent_distribution.errors import RuntimeActivationConflict, RuntimeActivationError
from intergrax.agent_distribution.stores import ApplicationEnvironmentServingStore
from intergrax.applications._shared.production_lifecycle import ProductionLifecycleServices
from intergrax.applications._shared.registry_projection import RegistryProjectionError
from intergrax.applications._shared.registry_projection_rehydrator import (
    RegistryProjectionRehydrationError,
    RegistryProjectionRehydrationResult,
    RuntimeRegistryProjectionRehydrator,
)

RECONCILIATION_OUTCOME_SKIPPED: Final = "reconciliation_skipped_no_traffic"
RECONCILIATION_OUTCOME_READY: Final = "reconciliation_ready"


class ProductionServingReconciliationError(Exception):
    """Production serving environment could not be reconciled after process restart."""


@dataclass(frozen=True, slots=True)
class ProductionServingReconciliationEvidence:
    """Inspectable evidence for one reconciliation attempt."""

    application_id: str
    application_environment_id: str
    runtime_revision_id: str | None
    serving_pointer_revision: int | None
    outcome: str
    physical_unit_live: bool | None = None
    attestation_evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ProductionServingReconciliationResult:
    """Outcome of one production serving reconciliation."""

    evidence: ProductionServingReconciliationEvidence
    attestation: ServingUnitAttestation | None
    rehydration: RegistryProjectionRehydrationResult | None


def reconcile_production_serving_environment_on_process_open(
    *,
    lifecycle_services: ProductionLifecycleServices,
    serving_store: ApplicationEnvironmentServingStore,
    registry_projection_rehydrator: RuntimeRegistryProjectionRehydrator,
    application_id: str,
    application_environment_id: str,
    expected_serving_pointer_revision: int | None = None,
) -> ProductionServingReconciliationResult:
    """Reconcile durable serving facts with live physical unit and projection inputs."""
    activation = lifecycle_services.activation_service
    try:
        attest_transition = (
            activation.reconcile_traffic_serving_physical_unit_after_process_restart(
                application_id=application_id,
                application_environment_id=application_environment_id,
                expected_serving_pointer_revision=expected_serving_pointer_revision,
            )
        )
    except (RuntimeActivationError, RuntimeActivationConflict) as exc:
        raise ProductionServingReconciliationError(str(exc)) from exc

    attestation = attest_transition.value
    serving = serving_store.get_serving_record(application_id, application_environment_id)
    if attestation is None:
        return ProductionServingReconciliationResult(
            evidence=ProductionServingReconciliationEvidence(
                application_id=application_id,
                application_environment_id=application_environment_id,
                runtime_revision_id=(
                    serving.traffic_serving_revision_id if serving is not None else None
                ),
                serving_pointer_revision=(
                    serving.serving_pointer_revision if serving is not None else None
                ),
                outcome=RECONCILIATION_OUTCOME_SKIPPED,
            ),
            attestation=None,
            rehydration=None,
        )

    revision_id = attestation.runtime_revision_id
    projection_input = lifecycle_services.projection_input_store.get(revision_id)
    if projection_input is None:
        raise ProductionServingReconciliationError(
            f"missing reconstructible projection inputs for {revision_id!r}",
        )

    try:
        rehydration = registry_projection_rehydrator.rehydrate_serving_registry_projection(
            application_id=application_id,
            application_environment_id=application_environment_id,
        )
    except RegistryProjectionRehydrationError as exc:
        raise ProductionServingReconciliationError(str(exc)) from exc

    try:
        lifecycle_services.projection_coordinator.prepare_projection(revision_id)
    except RegistryProjectionError as exc:
        raise ProductionServingReconciliationError(str(exc)) from exc

    pointer_revision = serving.serving_pointer_revision if serving is not None else None
    return ProductionServingReconciliationResult(
        evidence=ProductionServingReconciliationEvidence(
            application_id=application_id,
            application_environment_id=application_environment_id,
            runtime_revision_id=revision_id,
            serving_pointer_revision=pointer_revision,
            outcome=RECONCILIATION_OUTCOME_READY,
            physical_unit_live=attestation.physical_unit_live,
            attestation_evidence_ref=attestation.evidence_ref,
        ),
        attestation=attestation,
        rehydration=rehydration,
    )


__all__ = [
    "ProductionServingReconciliationError",
    "ProductionServingReconciliationEvidence",
    "ProductionServingReconciliationResult",
    "RECONCILIATION_OUTCOME_READY",
    "RECONCILIATION_OUTCOME_SKIPPED",
    "reconcile_production_serving_environment_on_process_open",
]
