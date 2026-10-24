# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical durable commit-authority-loss decision (TOOL-LIFE-X L3-R10)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason

CANONICAL_WORKER_EXECUTION_COMMIT_AUTHORITY_LOSS_DECISION_AUTHORITY_ID = (
    "tool-life-x.worker-execution-commit-authority-loss-decision.v1"
)


class WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError(RuntimeError):
    """Decision store write rejected — use sanctioned decision authority."""


class WorkerExecutionCommitAuthorityLossDecisionNotFoundError(RuntimeError):
    """No durable commit-authority-loss decision for the bound worker execution epoch."""


class WorkerExecutionCommitAuthorityLossDecisionRejectedError(RuntimeError):
    """Decision failed digest, freshness, or binding verification."""


class WorkerExecutionCommitAuthorityLossDecisionUnauthorizedError(RuntimeError):
    """Caller cannot issue or consume a commit-authority-loss decision for this binding."""


@dataclass(frozen=True, slots=True)
class WorkerExecutionCommitAuthorityLossDecisionV1:
    """Durable execution/recovery decision invalidating one worker execution commit epoch."""

    decision_id: str
    decision_authority_id: str
    tenant_id: str
    operation_id: str
    worker_instance_id: str
    ownership_epoch: int
    bound_store_revision: int
    loss_reason: CertifiedOwnerLossReason
    decided_at: datetime
    valid_until: datetime
    decision_digest: str

    def __post_init__(self) -> None:
        if type(self.loss_reason) is not CertifiedOwnerLossReason:
            raise TypeError("loss_reason must be CertifiedOwnerLossReason")


def compute_worker_execution_commit_authority_loss_decision_digest(
    decision: WorkerExecutionCommitAuthorityLossDecisionV1,
) -> str:
    payload = {
        "decision_id": decision.decision_id,
        "decision_authority_id": decision.decision_authority_id,
        "tenant_id": decision.tenant_id,
        "operation_id": decision.operation_id,
        "worker_instance_id": decision.worker_instance_id,
        "ownership_epoch": decision.ownership_epoch,
        "bound_store_revision": decision.bound_store_revision,
        "loss_reason": decision.loss_reason.value,
        "decided_at": decision.decided_at.astimezone(timezone.utc).isoformat(),
        "valid_until": decision.valid_until.astimezone(timezone.utc).isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@runtime_checkable
class WorkerExecutionCommitAuthorityLossDecisionPort(Protocol):
    """Durable ledger of canonical commit-authority-loss decisions."""

    def authoritative_commit_authority_loss_decision(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1 | None:
        ...

    def record_commit_authority_loss_decision(
        self,
        *,
        decision: WorkerExecutionCommitAuthorityLossDecisionV1,
    ) -> WorkerExecutionCommitAuthorityLossDecisionV1:
        ...


__all__ = [
    "CANONICAL_WORKER_EXECUTION_COMMIT_AUTHORITY_LOSS_DECISION_AUTHORITY_ID",
    "WorkerExecutionCommitAuthorityLossDecisionDirectInvocationError",
    "WorkerExecutionCommitAuthorityLossDecisionNotFoundError",
    "WorkerExecutionCommitAuthorityLossDecisionPort",
    "WorkerExecutionCommitAuthorityLossDecisionRejectedError",
    "WorkerExecutionCommitAuthorityLossDecisionUnauthorizedError",
    "WorkerExecutionCommitAuthorityLossDecisionV1",
    "compute_worker_execution_commit_authority_loss_decision_digest",
]
