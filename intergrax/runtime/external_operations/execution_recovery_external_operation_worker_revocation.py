# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned execution/recovery entry for recording worker execution epoch revocation.

Positive revocation requires a typed ``WorkerExecutionCommitAuthorityLossPolicy`` wired at
composition time. Production ``DURABLE_SINGLE_HOST`` composition defaults to
``FailClosedWorkerExecutionCommitAuthorityLossPolicy`` until the canonical execution/recovery
owner exposes an enforceable, durable commit-authority-loss decision (R8/R9 architecture
blocker — not a caller-supplied admission scope).
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationV1,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationRecorder,
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
    "ExecutionRecoveryExternalOperationWorkerRevocation",
]
