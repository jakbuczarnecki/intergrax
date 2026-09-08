# © Artur Czarnecki. All rights reserved.

"""MP-4D Approval service boundary — validation, authority invocation, domain orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from intergrax.approval._authority_enforcement import require_approval_allow
from intergrax.approval.authority_context import ApprovalAuthorityContextFactory
from intergrax.collaborative_work.enforcement_gate import (
    CollaborativeWorkEnforcementGate,
)
from intergrax.contracts.approval import (
    ApprovalLifecycleState,
    ApprovalRequest,
    CreateApprovalRequest,
    ExecuteHumanApprovalActionRequest,
    HumanApprovalAction,
)

TRUSTED_OPERATION_APPROVAL_CREATE: Final = "approval.request.create"
TRUSTED_OPERATION_APPROVAL_ACTION: Final = "approval.action.execute"


class ApprovalService:
    """Authority-aware Approval mutation boundary without persistence ownership."""

    def __init__(
        self,
        *,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._enforcement_gate = enforcement_gate
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_approval_request(
        self, request: CreateApprovalRequest
    ) -> ApprovalRequest:
        resource_scope = (
            ApprovalAuthorityContextFactory.create_resource_scope_for_create(request)
        )
        require_approval_allow(
            enforcement_gate=self._enforcement_gate,
            operation_id=TRUSTED_OPERATION_APPROVAL_CREATE,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._require_timezone_aware(self._clock())
        requested_by = request.requested_by_principal_id or request.acting_principal_id
        return ApprovalRequest(
            approval_id=request.approval_id,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            decision_id=request.decision_id,
            requested_by_principal_id=requested_by,
            requested_at=now,
            lifecycle_state=ApprovalLifecycleState.REQUESTED,
            references=request.references,
        )

    def execute_human_approval_action(
        self,
        *,
        existing_approval: ApprovalRequest,
        request: ExecuteHumanApprovalActionRequest,
    ) -> HumanApprovalAction:
        ApprovalAuthorityContextFactory.validate_action_scope_alignment(
            existing_approval=existing_approval,
            request=request,
        )
        resource_scope = (
            ApprovalAuthorityContextFactory.create_resource_scope_for_action(request)
        )
        require_approval_allow(
            enforcement_gate=self._enforcement_gate,
            operation_id=TRUSTED_OPERATION_APPROVAL_ACTION,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._require_timezone_aware(self._clock())
        return HumanApprovalAction(
            approval_id=existing_approval.approval_id,
            acting_principal_id=request.acting_principal_id,
            action=request.action,
            timestamp=now,
            comment_reference=request.comment_reference,
        )

    @staticmethod
    def _require_timezone_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return value
