# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconcile-before-pin orchestration for P2 pin records (TRACE-X-P5-R2-P4-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    IntegrationConfigurationSubject,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
    find_pin_record_for_subject,
    obligation_requires_recovery_staging,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
    ExecutionIntegrationConfigurationPinningFailureReason,
    ExecutionIntegrationConfigurationPinningStore,
    validate_pin_subject_against_provenance,
)


def reconcile_requirement_recovery_staging_for_pin(
    *,
    pinning_store: ExecutionIntegrationConfigurationPinningStore,
    tenant_id: str,
    execution_id: ExecutionId,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
    candidate_staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging | None,
) -> ExecutionIntegrationConfigurationRequirementRecoveryStaging | None:
    """
    Return staging to pass to ``pin()`` after reconcile-before-pin read.

    Reuses stored staging when a durable row exists; requires staging for new
    CONFIGURED_ADOPTED obligations when no row exists.
    """
    validate_pin_subject_against_provenance(subject=subject, provenance=provenance)
    records = pinning_store.read_pin_records(
        tenant_id=tenant_id,
        execution_id=execution_id,
    )
    existing = find_pin_record_for_subject(records, subject)
    if existing is not None:
        if existing.provenance != provenance:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CONFLICT,
                detail="provenance mismatch on reconcile",
            )
        staging = existing.requirement_recovery_staging
        if obligation_requires_recovery_staging(provenance) and staging is None:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.RECOVERY_STAGING_UNAVAILABLE,
            )
        return staging
    if obligation_requires_recovery_staging(provenance) and candidate_staging is None:
        raise ExecutionIntegrationConfigurationPinningError(
            ExecutionIntegrationConfigurationPinningFailureReason.RECOVERY_STAGING_UNAVAILABLE,
            detail="staging required for first CONFIGURED_ADOPTED pin",
        )
    return candidate_staging


def pin_with_reconcile(
    *,
    pinning_store: ExecutionIntegrationConfigurationPinningStore,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
    candidate_staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging | None,
) -> None:
    staging = reconcile_requirement_recovery_staging_for_pin(
        pinning_store=pinning_store,
        tenant_id=provenance.tenant_id,
        execution_id=provenance.execution_id,
        subject=subject,
        provenance=provenance,
        candidate_staging=candidate_staging,
    )
    pinning_store.pin(
        subject=subject,
        provenance=provenance,
        requirement_recovery_staging=staging,
    )


__all__ = [
    "pin_with_reconcile",
    "reconcile_requirement_recovery_staging_for_pin",
]
