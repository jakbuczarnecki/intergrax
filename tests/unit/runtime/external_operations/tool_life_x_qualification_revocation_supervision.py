# © Artur Czarnecki. All rights reserved.

"""L3-R7/R8 qualification-only revocation supervision (not production composition)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    mint_worker_execution_ownership_revocation_authorization,
)


@dataclass(frozen=True, slots=True)
class QualificationWorkerExecutionRevocationSupervision:
    """Explicit test double for supervision-gated revocation authorization."""

    def authorize_worker_execution_ownership_revocation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        bound_store_revision: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> WorkerExecutionOwnershipRevocationAuthorizationV1:
        return mint_worker_execution_ownership_revocation_authorization(
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=bound_store_revision,
            loss_reason=loss_reason,
        )
