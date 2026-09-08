# © Artur Czarnecki. All rights reserved.

"""HITL approval boundary with persistent approval state."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.agent_runtime_governance import (
    ApprovalRequest,
    ApprovalRequestStatus,
    ApprovalStorePort,
    ToolAuthorizationRequest,
)


class InMemoryApprovalStore:
    """In-process approval store for governed continuation."""

    def __init__(self) -> None:
        self._store: dict[str, ApprovalRequest] = {}

    def create(self, approval: ApprovalRequest) -> ApprovalRequest:
        self._store[approval.approval_id] = approval
        return approval

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._store.get(approval_id)

    def update(self, approval: ApprovalRequest) -> ApprovalRequest:
        if approval.approval_id not in self._store:
            raise KeyError(f"approval not found: {approval.approval_id}")
        self._store[approval.approval_id] = approval
        return approval


def mint_approval_id() -> str:
    return f"approval_{uuid4().hex}"


class AgentRuntimeApprovalBoundary:
    """
    Asynchronous approval contract — no blocking inside policy engine.

    Approval lifecycle: CREATED → WAITING_FOR_APPROVAL → APPROVED | REJECTED | EXPIRED
    """

    def __init__(self, store: ApprovalStorePort) -> None:
        self._store = store

    def create_approval_request(
        self,
        authorization_request: ToolAuthorizationRequest,
        *,
        expires_at: datetime,
        audit_metadata: tuple[str, ...] = (),
    ) -> ApprovalRequest:
        now = datetime.now(timezone.utc)
        approval = ApprovalRequest(
            approval_id=mint_approval_id(),
            authorization_request=authorization_request,
            status=ApprovalRequestStatus.WAITING_FOR_APPROVAL,
            risk_classification=authorization_request.risk_level,
            requested_at=now,
            expires_at=expires_at,
            audit_metadata=audit_metadata,
        )
        return self._store.create(approval)

    def approve(
        self,
        approval_id: str,
        *,
        decided_by: str,
        decided_at: datetime | None = None,
    ) -> ApprovalRequest:
        existing = self._require_pending(approval_id)
        now = decided_at or datetime.now(timezone.utc)
        if now >= existing.expires_at:
            return self._expire(existing, now=now)
        updated = existing.model_copy(
            update={
                "status": ApprovalRequestStatus.APPROVED,
                "decided_at": now,
                "decided_by": decided_by,
            },
        )
        return self._store.update(updated)

    def reject(
        self,
        approval_id: str,
        *,
        decided_by: str,
        decided_at: datetime | None = None,
    ) -> ApprovalRequest:
        existing = self._require_pending(approval_id)
        now = decided_at or datetime.now(timezone.utc)
        updated = existing.model_copy(
            update={
                "status": ApprovalRequestStatus.REJECTED,
                "decided_at": now,
                "decided_by": decided_by,
            },
        )
        return self._store.update(updated)

    def check_expiration(self, approval_id: str) -> ApprovalRequest:
        existing = self._store.get(approval_id)
        if existing is None:
            raise KeyError(f"approval not found: {approval_id}")
        if existing.status is not ApprovalRequestStatus.WAITING_FOR_APPROVAL:
            return existing
        now = datetime.now(timezone.utc)
        if now >= existing.expires_at:
            return self._expire(existing, now=now)
        return existing

    def is_approved(self, approval_id: str) -> bool:
        record = self.check_expiration(approval_id)
        return record.status is ApprovalRequestStatus.APPROVED

    def _require_pending(self, approval_id: str) -> ApprovalRequest:
        existing = self.check_expiration(approval_id)
        if existing.status is not ApprovalRequestStatus.WAITING_FOR_APPROVAL:
            raise ValueError(
                f"approval {approval_id} is not pending (status={existing.status.value})"
            )
        return existing

    def _expire(self, existing: ApprovalRequest, *, now: datetime) -> ApprovalRequest:
        updated = existing.model_copy(
            update={"status": ApprovalRequestStatus.EXPIRED},
        )
        return self._store.update(updated)
