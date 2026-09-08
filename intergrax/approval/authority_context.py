# © Artur Czarnecki. All rights reserved.

"""Approval authority context factory — scope alignment without policy ownership."""

from __future__ import annotations

from intergrax.approval.errors import ApprovalAuthorityContextError
from intergrax.contracts.approval import (
    ApprovalRequest,
    CreateApprovalRequest,
    ExecuteHumanApprovalActionRequest,
    approval_resource_scope,
)


class ApprovalAuthorityContextFactory:
    """Map Approval mutation requests to deterministic MP-1 resource scopes."""

    @staticmethod
    def create_resource_scope_for_create(request: CreateApprovalRequest) -> str:
        return approval_resource_scope(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            decision_id=request.decision_id,
        )

    @staticmethod
    def create_resource_scope_for_action(
        request: ExecuteHumanApprovalActionRequest,
    ) -> str:
        return approval_resource_scope(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            approval_id=request.approval_id,
        )

    @staticmethod
    def validate_action_scope_alignment(
        *,
        existing_approval: ApprovalRequest,
        request: ExecuteHumanApprovalActionRequest,
    ) -> None:
        if existing_approval.approval_id != request.approval_id:
            raise ApprovalAuthorityContextError(
                "approval_id must match the existing approval request",
            )
        if existing_approval.tenant_id != request.tenant_id:
            raise ApprovalAuthorityContextError(
                "tenant_id must match the existing approval request",
            )
        if existing_approval.workspace_id != request.workspace_id:
            raise ApprovalAuthorityContextError(
                "workspace_id must match the existing approval request",
            )
