# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical commit-authority-loss decision authority and durable loss policy (L3-R10)."""

from __future__ import annotations

import contextlib
import contextvars
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.external_operation_cancellation import ExternalOperationNotFoundError
from intergrax.contracts.worker_execution_commit_authority_loss_decision import (
    CANONICAL_WORKER_EXECUTION_COMMIT_AUTHORITY_LOSS_DECISION_AUTHORITY_ID,
    WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError,
    WorkerExecutionCommitAuthorityLossDecisionNotFoundError,
    WorkerExecutionCommitAuthorityLossDecisionPort,
    WorkerExecutionCommitAuthorityLossDecisionRejectedError,
    WorkerExecutionCommitAuthorityLossDecisionUnauthorizedError,
    WorkerExecutionCommitAuthorityLossDecisionV1,
    compute_worker_execution_commit_authority_loss_decision_digest,
)
from intergrax.contracts.worker_execution_ownership_revocation import (
    WorkerExecutionOwnershipRevocationUnauthorizedError,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.worker_execution_ownership_revocation import (
    assert_authoritative_worker_execution_revocation_allowed,
)

_DECISION_TTL = timedelta(minutes=30)

_decision_authority_invocation_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "worker_execution_commit_authority_loss_decision_authority_invocation",
    default=False,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextlib.contextmanager
def _decision_authority_invocation():
    token = _decision_authority_invocation_flag.set(True)
    try:
        yield
    finally:
        _decision_authority_invocation_flag.reset(token)


def _require_decision_authority_invocation() -> None:
    if not _decision_authority_invocation_flag.get():
        raise WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError(
            "use WorkerExecutionCommitAuthorityLossDecisionAuthority",
        )


def verify_worker_execution_commit_authority_loss_decision(
    decision: WorkerExecutionCommitAuthorityLossDecisionV1,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    bound_store_revision: int,
    loss_reason: CertifiedOwnerLossReason,
) -> None:
    expected = compute_worker_execution_commit_authority_loss_decision_digest(decision)
    if decision.decision_digest != expected:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "commit-authority-loss decision digest mismatch",
        )
    if decision.decision_authority_id != (
        CANONICAL_WORKER_EXECUTION_COMMIT_AUTHORITY_LOSS_DECISION_AUTHORITY_ID
    ):
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "commit-authority-loss decision authority mismatch",
        )
    if decision.tenant_id != tenant_id:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "decision tenant mismatch",
        )
    if decision.operation_id != operation_id:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "decision operation mismatch",
        )
    if decision.worker_instance_id != worker_instance_id:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "decision worker_instance_id mismatch",
        )
    if decision.ownership_epoch != ownership_epoch:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "decision ownership_epoch mismatch",
        )
    if decision.bound_store_revision != bound_store_revision:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "stale commit-authority-loss decision for store revision",
        )
    if decision.loss_reason != loss_reason:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "decision loss_reason mismatch",
        )
    now = _utc_now()
    if decision.valid_until < now:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "commit-authority-loss decision expired",
        )
    if decision.decided_at > now:
        raise WorkerExecutionCommitAuthorityLossDecisionRejectedError(
            "commit-authority-loss decision not yet valid",
        )


def require_authoritative_commit_authority_loss_decision(
    port: WorkerExecutionCommitAuthorityLossDecisionPort,
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    bound_store_revision: int,
    loss_reason: CertifiedOwnerLossReason,
) -> WorkerExecutionCommitAuthorityLossDecisionV1:
    decision = port.authoritative_commit_authority_loss_decision(
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
    )
    if decision is None:
        raise WorkerExecutionCommitAuthorityLossDecisionNotFoundError(
            "no durable commit-authority-loss decision for bound epoch",
        )
    verify_worker_execution_commit_authority_loss_decision(
        decision,
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
        bound_store_revision=bound_store_revision,
        loss_reason=loss_reason,
    )
    return decision


@dataclass(frozen=True, slots=True)
class WorkerExecutionCommitAuthorityLossDecisionAuthority:
    """Sanctioned issuer of durable commit-authority-loss decisions for bound operations."""

    authoritative_state_store: ExternalOperationStateStore
    decision_port: WorkerExecutionCommitAuthorityLossDecisionPort

    def issue_for_bound_operation(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        loss_reason: CertifiedOwnerLossReason,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1:
        loaded = self.authoritative_state_store.load(tenant_id, operation_id)
        if loaded is None:
            raise ExternalOperationNotFoundError(operation_id)
        if loaded.worker_instance_id is None or loaded.worker_ownership_epoch is None:
            raise WorkerExecutionOwnershipRevocationUnauthorizedError(
                "worker execution ownership is not bound on authoritative record",
            )
        assert_authoritative_worker_execution_revocation_allowed(
            loaded,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=loaded.worker_instance_id,
            ownership_epoch=loaded.worker_ownership_epoch,
        )
        existing = self.decision_port.authoritative_commit_authority_loss_decision(
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=loaded.worker_instance_id,
            ownership_epoch=loaded.worker_ownership_epoch,
        )
        if existing is not None:
            verify_worker_execution_commit_authority_loss_decision(
                existing,
                tenant_id=tenant_id,
                operation_id=operation_id,
                worker_instance_id=loaded.worker_instance_id,
                ownership_epoch=loaded.worker_ownership_epoch,
                bound_store_revision=loaded.revision,
                loss_reason=loss_reason,
            )
            return existing
        when = _utc_now()
        decision_id = f"wexec-commit-loss:{uuid.uuid4().hex}"
        provisional = WorkerExecutionCommitAuthorityLossDecisionV1(
            decision_id=decision_id,
            decision_authority_id=(
                CANONICAL_WORKER_EXECUTION_COMMIT_AUTHORITY_LOSS_DECISION_AUTHORITY_ID
            ),
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=loaded.worker_instance_id,
            ownership_epoch=loaded.worker_ownership_epoch,
            bound_store_revision=loaded.revision,
            loss_reason=loss_reason,
            decided_at=when,
            valid_until=when + _DECISION_TTL,
            decision_digest="pending",
        )
        digest = compute_worker_execution_commit_authority_loss_decision_digest(provisional)
        decision = WorkerExecutionCommitAuthorityLossDecisionV1(
            decision_id=decision_id,
            decision_authority_id=provisional.decision_authority_id,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=loaded.worker_instance_id,
            ownership_epoch=loaded.worker_ownership_epoch,
            bound_store_revision=loaded.revision,
            loss_reason=loss_reason,
            decided_at=when,
            valid_until=provisional.valid_until,
            decision_digest=digest,
        )
        with _decision_authority_invocation():
            return self.decision_port.record_commit_authority_loss_decision(
                decision=decision,
            )


@dataclass(frozen=True, slots=True)
class DurableWorkerExecutionCommitAuthorityLossPolicy:
    """Production loss policy — requires a durable canonical decision at current store revision."""

    decision_port: WorkerExecutionCommitAuthorityLossDecisionPort
    authoritative_state_store: ExternalOperationStateStore

    def require_commit_authority_loss_admitted(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        loss_reason: CertifiedOwnerLossReason,
    ) -> None:
        record = self.authoritative_state_store.load(tenant_id, operation_id)
        if record is None:
            raise WorkerExecutionCommitAuthorityLossDecisionUnauthorizedError(
                "authoritative operation record missing for loss admission",
            )
        require_authoritative_commit_authority_loss_decision(
            self.decision_port,
            tenant_id=tenant_id,
            operation_id=operation_id,
            worker_instance_id=worker_instance_id,
            ownership_epoch=ownership_epoch,
            bound_store_revision=record.revision,
            loss_reason=loss_reason,
        )


class InMemoryWorkerExecutionCommitAuthorityLossDecisionStore(
    WorkerExecutionCommitAuthorityLossDecisionPort,
):
    """Process-local decision ledger for qualification and unit tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[
            tuple[str, str, str, int],
            WorkerExecutionCommitAuthorityLossDecisionV1,
        ] = {}

    def authoritative_commit_authority_loss_decision(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1 | None:
        key = (tenant_id, operation_id, worker_instance_id, ownership_epoch)
        with self._lock:
            return self._records.get(key)

    def record_commit_authority_loss_decision(
        self,
        *,
        decision: WorkerExecutionCommitAuthorityLossDecisionV1,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1:
        _require_decision_authority_invocation()
        key = (
            decision.tenant_id,
            decision.operation_id,
            decision.worker_instance_id,
            decision.ownership_epoch,
        )
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            verify_worker_execution_commit_authority_loss_decision(
                decision,
                tenant_id=decision.tenant_id,
                operation_id=decision.operation_id,
                worker_instance_id=decision.worker_instance_id,
                ownership_epoch=decision.ownership_epoch,
                bound_store_revision=decision.bound_store_revision,
                loss_reason=decision.loss_reason,
            )
            self._records[key] = decision
            return decision


__all__ = [
    "DurableWorkerExecutionCommitAuthorityLossPolicy",
    "InMemoryWorkerExecutionCommitAuthorityLossDecisionStore",
    "WorkerExecutionCommitAuthorityLossDecisionAuthority",
    "require_authoritative_commit_authority_loss_decision",
    "verify_worker_execution_commit_authority_loss_decision",
]
