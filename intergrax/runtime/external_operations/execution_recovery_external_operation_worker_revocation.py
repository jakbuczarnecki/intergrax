# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned execution/recovery entry for recording worker execution epoch revocation."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    WorkerExecutionOwnershipRevocationV1,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationRecorder,
)

_COMMIT_AUTHORITY_LOSS_ADMISSION: contextvars.ContextVar[
    tuple[str, str, str, int, CertifiedOwnerLossReason] | None
] = contextvars.ContextVar(
    "worker_execution_commit_authority_loss_admission",
    default=None,
)


@contextlib.contextmanager
def admit_worker_execution_commit_authority_loss(
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    loss_reason: CertifiedOwnerLossReason,
):
    token = _COMMIT_AUTHORITY_LOSS_ADMISSION.set(
        (tenant_id, operation_id, worker_instance_id, ownership_epoch, loss_reason),
    )
    try:
        yield
    finally:
        _COMMIT_AUTHORITY_LOSS_ADMISSION.reset(token)


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryAdmittedCommitAuthorityLossPolicy:
    """Admits revocation only while execution/recovery holds an explicit admission scope."""

    def require_commit_authority_loss_admitted(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> None:
        admitted = _COMMIT_AUTHORITY_LOSS_ADMISSION.get()
        expected = (
            tenant_id,
            operation_id,
            worker_instance_id,
            ownership_epoch,
            loss_reason,
        )
        if admitted != expected:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "worker execution commit authority loss admission scope mismatch",
            )


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryExternalOperationWorkerRevocation:
    """Canonical execution/recovery owner surface for durable worker epoch revocation."""

    recorder: WorkerExecutionOwnershipRevocationRecorder

    def record_revocation_after_commit_authority_loss(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
    ) -> WorkerExecutionOwnershipRevocationV1:
        del worker_instance_id, ownership_epoch
        return self.recorder.record_revocation_for_bound_operation(
            tenant_id=tenant_id,
            operation_id=operation_id,
            loss_reason=loss_reason,
        )


__all__ = [
    "ExecutionRecoveryAdmittedCommitAuthorityLossPolicy",
    "ExecutionRecoveryExternalOperationWorkerRevocation",
    "admit_worker_execution_commit_authority_loss",
]
