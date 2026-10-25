# © Artur Czarnecki. All rights reserved.

"""L3-R7/R8/R9 qualification-only revocation supervision (not production composition)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.runtime.external_operations.execution_recovery_worker_execution_revocation_supervision import (
    ExecutionRecoveryWorkerExecutionRevocationSupervision,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    WorkerExecutionOwnershipRevocationTrustMaterial,
    bind_worker_execution_ownership_revocation_trust_for_qualification,
)


@dataclass(frozen=True, slots=True)
class QualificationWorkerExecutionCommitAuthorityLossPolicy:
    """Test-only policy — never use in production composition."""

    def require_commit_authority_loss_admitted(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> None:
        return


def qualification_worker_execution_revocation_supervision() -> (
    ExecutionRecoveryWorkerExecutionRevocationSupervision
):
    bind_worker_execution_ownership_revocation_trust_for_qualification()
    return ExecutionRecoveryWorkerExecutionRevocationSupervision(
        loss_policy=QualificationWorkerExecutionCommitAuthorityLossPolicy(),
        trust=WorkerExecutionOwnershipRevocationTrustMaterial(
            mac_key=b"tool-life-x-qualification-revocation-mac-v1",
        ),
    )


class QualificationWorkerExecutionRevocationSupervision:
    """Explicit test double delegating to sanctioned execution/recovery supervision."""

    def __init__(self) -> None:
        self._inner = qualification_worker_execution_revocation_supervision()

    def authorize_worker_execution_ownership_revocation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        bound_store_revision: int,
        loss_reason: CertifiedOwnerLossReason,
    ):
        return self._inner.authorize_worker_execution_ownership_revocation(
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=bound_store_revision,
            loss_reason=loss_reason,
        )
