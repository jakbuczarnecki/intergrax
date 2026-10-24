# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution/recovery supervision adapter for worker execution epoch revocation (L3-R9)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_ownership_revocation import (
    CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    WorkerExecutionOwnershipRevocationSupervisionPort,
    WorkerExecutionOwnershipRevocationUnauthorizedError,
    compute_worker_execution_ownership_revocation_authorization_digest,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation_trust import (
    WorkerExecutionOwnershipRevocationTrustMaterial,
    compute_worker_execution_ownership_revocation_authorization_mac,
    require_worker_execution_ownership_revocation_trust,
)

_AUTHORIZATION_TTL = timedelta(minutes=5)


@runtime_checkable
class WorkerExecutionCommitAuthorityLossPolicy(Protocol):
    """Execution/recovery plane — admits loss of worker commit authority for one bound epoch."""

    def require_commit_authority_loss_admitted(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class FailClosedWorkerExecutionCommitAuthorityLossPolicy:
    """Production default until execution/recovery admits a sanctioned loss observation."""

    def require_commit_authority_loss_admitted(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> None:
        raise WorkerExecutionOwnershipRevocationUnauthorizedError(
            "worker execution commit authority loss is not admitted by execution/recovery",
        )


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryWorkerExecutionRevocationSupervision:
    """Canonical execution/recovery supervision — typed MAC decisions, fail-closed policy."""

    loss_policy: WorkerExecutionCommitAuthorityLossPolicy
    trust: WorkerExecutionOwnershipRevocationTrustMaterial | None = None

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
        self.loss_policy.require_commit_authority_loss_admitted(
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            loss_reason=loss_reason,
        )
        trust = self.trust or require_worker_execution_ownership_revocation_trust()
        when = datetime.now(timezone.utc)
        authorization_id = f"wexec-revoke-auth:{uuid.uuid4().hex}"
        provisional = WorkerExecutionOwnershipRevocationAuthorizationV1(
            authorization_id=authorization_id,
            supervision_authority_id=(
                CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID
            ),
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=bound_store_revision,
            loss_reason=loss_reason,
            authorized_at=when,
            valid_until=when + _AUTHORIZATION_TTL,
            authorization_digest="pending",
            authorization_mac="pending",
        )
        digest = compute_worker_execution_ownership_revocation_authorization_digest(
            provisional,
        )
        with_digest = WorkerExecutionOwnershipRevocationAuthorizationV1(
            authorization_id=authorization_id,
            supervision_authority_id=provisional.supervision_authority_id,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=bound_store_revision,
            loss_reason=loss_reason,
            authorized_at=when,
            valid_until=provisional.valid_until,
            authorization_digest=digest,
            authorization_mac="pending",
        )
        mac = compute_worker_execution_ownership_revocation_authorization_mac(
            with_digest,
            trust=trust,
        )
        return WorkerExecutionOwnershipRevocationAuthorizationV1(
            authorization_id=authorization_id,
            supervision_authority_id=provisional.supervision_authority_id,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=bound_store_revision,
            loss_reason=loss_reason,
            authorized_at=when,
            valid_until=provisional.valid_until,
            authorization_digest=digest,
            authorization_mac=mac,
        )


__all__ = [
    "ExecutionRecoveryWorkerExecutionRevocationSupervision",
    "FailClosedWorkerExecutionCommitAuthorityLossPolicy",
    "WorkerExecutionCommitAuthorityLossPolicy",
]
