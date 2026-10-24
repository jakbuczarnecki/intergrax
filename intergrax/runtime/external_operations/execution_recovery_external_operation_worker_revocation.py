# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned execution/recovery entry for recording worker execution epoch revocation.

Positive revocation requires execution/recovery supervision admission and records a durable
``WorkerExecutionCommitAuthorityLossDecisionV1`` atomically with epoch revocation (L3-R10).
Callers must not synthesize loss decisions outside that authoritative transition.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_commit_authority_loss_decision import (
    WorkerExecutionCommitAuthorityLossDecisionV1,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationV1,
)
from intergrax.runtime.external_operations.worker_execution_commit_authority_loss import (
    WorkerExecutionCommitAuthorityLossDecisionAuthority,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationRecorder,
)


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryExternalOperationWorkerRevocation:
    """Canonical execution/recovery owner surface for durable worker epoch revocation."""

    recorder: WorkerExecutionOwnershipRevocationRecorder
    decision_authority: WorkerExecutionCommitAuthorityLossDecisionAuthority

    def decide_commit_authority_loss_for_bound_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason = CertifiedOwnerLossReason.WORKER_LOST,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1:
        return self.decision_authority.issue_for_bound_operation(
            tenant_id=tenant_id,
            operation_id=operation_id,
            loss_reason=loss_reason,
        )

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
