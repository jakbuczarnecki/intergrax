# © Artur Czarnecki. All rights reserved.

"""Adversarial authorization forgery helpers (tests only)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from intergrax.contracts.certified_owner_loss_evidence import CertifiedOwnerLossReason
from intergrax.contracts.worker_execution_ownership_revocation import (
    CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
    WorkerExecutionOwnershipRevocationAuthorizationV1,
    compute_worker_execution_ownership_revocation_authorization_digest,
)


def forged_worker_execution_ownership_revocation_authorization(
    *,
    tenant_id: str,
    operation_id: str,
    worker_instance_id: str,
    ownership_epoch: int,
    bound_store_revision: int,
    loss_reason: CertifiedOwnerLossReason,
    authorization_mac: str = "0" * 64,
) -> WorkerExecutionOwnershipRevocationAuthorizationV1:
    when = datetime.now(timezone.utc)
    authorization_id = f"wexec-revoke-auth-forged:{uuid.uuid4().hex}"
    provisional = WorkerExecutionOwnershipRevocationAuthorizationV1(
        authorization_id=authorization_id,
        supervision_authority_id=CANONICAL_WORKER_EXECUTION_REVOCATION_SUPERVISION_AUTHORITY_ID,
        tenant_id=tenant_id,
        operation_id=operation_id,
        worker_instance_id=worker_instance_id,
        ownership_epoch=ownership_epoch,
        bound_store_revision=bound_store_revision,
        loss_reason=loss_reason,
        authorized_at=when,
        valid_until=when + timedelta(minutes=5),
        authorization_digest="pending",
        authorization_mac=authorization_mac,
    )
    digest = compute_worker_execution_ownership_revocation_authorization_digest(
        provisional,
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
        authorization_mac=authorization_mac,
    )
