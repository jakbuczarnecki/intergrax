# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery admission gate for unknown external operations (W4-C + W3)."""

from __future__ import annotations

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationPhysicalState,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
)


class ExternalOperationRecoveryBlockedError(RuntimeError):
    """Recovery cannot proceed while external operation physical state is UNKNOWN."""


class ExternalOperationReconciliationAdmissionError(RuntimeError):
    """UNKNOWN external operation cannot enter reconciliation admission."""


def prepare_external_operations_for_recovery(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    active_owner: ProcessLocalExternalOperationOwner,
    operation_ids: tuple[str, ...] = (),
) -> None:
    """Fail closed when scoped operations are not terminal; owner-loss recovery is explicit."""
    del active_owner
    for operation_id in operation_ids:
        record = store.load(tenant_id, operation_id)
        if record is None:
            continue
        if record.physical_state is ExternalOperationPhysicalState.UNKNOWN:
            raise ExternalOperationRecoveryBlockedError(
                f"external operation {operation_id!r} is UNKNOWN; reconcile first",
            )
        if record.physical_state is ExternalOperationPhysicalState.RUNNING:
            raise ExternalOperationRecoveryBlockedError(
                f"external operation {operation_id!r} still RUNNING",
            )


def admit_unknown_external_operation_for_reconciliation(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
) -> None:
    """Certified loss must have fenced RUNNING→UNKNOWN before reconciliation admission."""
    record = store.load(tenant_id, operation_id)
    if record is None:
        raise ExternalOperationReconciliationAdmissionError(
            f"external operation {operation_id!r} not found",
        )
    if record.physical_state is not ExternalOperationPhysicalState.UNKNOWN:
        raise ExternalOperationReconciliationAdmissionError(
            f"external operation {operation_id!r} is not UNKNOWN for reconciliation",
        )
