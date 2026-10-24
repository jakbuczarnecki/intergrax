# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative worker execution ownership revocation (TOOL-LIFE-X L3-R7).

Generic ``WorkerInstance.revision`` and lifecycle labels are not ownership epochs.
Certified owner-loss requires a durable, fenced revocation record from this port.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason


class CertifiedOwnerLossWorkerOwnershipIndeterminateError(RuntimeError):
    """Worker authority cannot be read reliably — certification denied (fail-closed)."""


class WorkerExecutionOwnershipNotRevokedError(RuntimeError):
    """Bound execution ownership epoch has no authoritative revocation on record."""


class WorkerExecutionOwnershipRevocationUnauthorizedError(RuntimeError):
    """Revocation rejected — caller identity does not match authoritative ownership."""


class WorkerExecutionDirectRevocationInvocationError(RuntimeError):
    """Direct revocation port write rejected — use sanctioned composition recorder."""


class WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError(RuntimeError):
    """Revocation supervision port is not wired — positive revocation remains fail-closed."""


class WorkerExecutionOwnershipRevocationAuthorizationRejectedError(RuntimeError):
    """Typed revocation authorization failed verification at store boundary."""


CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID = (
    "tool-life-x.worker-execution-revocation-supervision.v1"
)


def worker_execution_revocation_proof_ref(
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    revocation_id: str,
) -> str:
    return (
        f"wexec-revoke:v1:{tenant_id}:{operation_id}:{worker_instance_id}"
        f":{ownership_epoch}:{revocation_id}"
    )


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationAuthorizationV1:
    """Governance-separated decision binding one durable worker execution epoch."""

    authorization_id: str
    supervision_authority_id: str
    tenant_id: str
    operation_id: str
    worker_instance_id: str
    ownership_epoch: int
    bound_store_revision: int
    loss_reason: CertifiedOwnerLossReason
    authorized_at: datetime
    authorization_digest: str

    def __post_init__(self) -> None:
        if type(self.loss_reason) is not CertifiedOwnerLossReason:
            raise TypeError("loss_reason must be CertifiedOwnerLossReason")


def compute_worker_execution_ownership_revocation_authorization_digest(
    authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
) -> str:
    payload = {
        "authorization_id": authorization.authorization_id,
        "supervision_authority_id": authorization.supervision_authority_id,
        "tenant_id": authorization.tenant_id,
        "operation_id": authorization.operation_id,
        "worker_instance_id": authorization.worker_instance_id,
        "ownership_epoch": authorization.ownership_epoch,
        "bound_store_revision": authorization.bound_store_revision,
        "loss_reason": authorization.loss_reason.value,
        "authorized_at": authorization.authorized_at.isoformat(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mint_worker_execution_ownership_revocation_authorization(
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    bound_store_revision: int,
    loss_reason: CertifiedOwnerLossReason,
    supervision_authority_id: str = CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
    authorized_at: datetime | None = None,
) -> WorkerExecutionOwnershipRevocationAuthorizationV1:
    when = authorized_at or datetime.now(timezone.utc)
    authorization_id = f"wexec-revoke-auth:{uuid.uuid4().hex}"
    provisional = WorkerExecutionOwnershipRevocationAuthorizationV1(
        authorization_id=authorization_id,
        supervision_authority_id=supervision_authority_id,
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
        bound_store_revision=bound_store_revision,
        loss_reason=loss_reason,
        authorized_at=when,
        authorization_digest="pending",
    )
    digest = compute_worker_execution_ownership_revocation_authorization_digest(
        provisional,
    )
    return WorkerExecutionOwnershipRevocationAuthorizationV1(
        authorization_id=authorization_id,
        supervision_authority_id=supervision_authority_id,
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
        bound_store_revision=bound_store_revision,
        loss_reason=loss_reason,
        authorized_at=when,
        authorization_digest=digest,
    )


@runtime_checkable
class WorkerExecutionOwnershipRevocationSupervisionPort(Protocol):
    """Sanctioned supervision plane — distinct from store CAS and issuer allowlist."""

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
        """Mint a typed revocation decision for the bound epoch (composition-only)."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerExecutionOwnershipRevocationV1:
    tenant_id: str
    operation_id: str
    worker_instance_id: str
    ownership_epoch: int
    revocation_id: str
    revoked_at: datetime
    proof_ref: str


@runtime_checkable
class WorkerExecutionOwnershipRevocationPort(Protocol):
    """Single semantic authority for fenced worker execution ownership revocation."""

    def authoritative_revocation_proof_ref(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
    ) -> str | None:
        """Return opaque proof ref when revocation is durable; else ``None``."""
        ...

    def revoke_worker_execution_ownership(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        worker_instance_id: str,
        ownership_epoch: int,
        authorization: WorkerExecutionOwnershipRevocationAuthorizationV1,
    ) -> WorkerExecutionOwnershipRevocationV1:
        """Record authoritative revocation for the bound epoch (composition-only)."""
        ...
