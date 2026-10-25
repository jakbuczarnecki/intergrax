# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TL8 sanctioned redrive authority transfer after reconciliation admission."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
    assert_redrive_effect_class_claim_matches_authoritative,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    ExternalOperationReconciliationAdmissionError,
    admit_unknown_external_operation_for_reconciliation,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationState


def _redrive_admission_id(
    *,
    tenant_id: str,
    operation_id: str,
    store_revision: int,
) -> str:
    return f"redrive-admit:{tenant_id}:{operation_id}:rev{store_revision}"


def transfer_redrive_authority_after_reconciliation(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    trusted_effect_contract: ExternalEffectContract,
    worker_instance_id: str,
    worker_ownership_epoch: int,
    caller_claimed_effect_class: ExternalOperationRedriveEffectClass | None = None,
) -> ExternalOperationState:
    """Atomic durable authority transfer for eligible Class B effects (qual + controlled)."""
    authoritative = resolve_authoritative_redrive_effect_class(trusted_effect_contract)
    if caller_claimed_effect_class is not None:
        assert_redrive_effect_class_claim_matches_authoritative(
            caller_claimed_effect_class=caller_claimed_effect_class,
            authoritative_effect_class=authoritative,
        )
    if authoritative is not ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED:
        raise ExternalOperationRedriveEffectClassRejectedError(
            f"trusted contract {trusted_effect_contract.contract_id!r} is not eligible "
            f"for automated redrive ({authoritative.value})",
        )
    admit_unknown_external_operation_for_reconciliation(
        store,
        tenant_id=tenant_id,
        operation_id=operation_id,
    )
    admission_id = _redrive_admission_id(
        tenant_id=tenant_id,
        operation_id=operation_id,
        store_revision=expected_revision,
    )
    return store.transfer_redrive_worker_execution_authority(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        worker_instance_id=worker_instance_id,
        worker_ownership_epoch=worker_ownership_epoch,
        redrive_admission_id=admission_id,
    )


__all__ = [
    "ExternalOperationReconciliationAdmissionError",
    "transfer_redrive_authority_after_reconciliation",
]
