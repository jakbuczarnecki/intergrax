# © Artur Czarnecki. All rights reserved.

"""Scoped approval consumption for control-plane mutation continuation (CLA-04)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationApprovalGrant,
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationDenialRecord,
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision


class ControlPlaneMutationApprovalError(ValueError):
    """Fail-closed control-plane mutation approval without canonical scope."""


def matches_authorization_scope(
    grant: ControlPlaneMutationApprovalGrant,
    scope: ControlPlaneMutationAuthorizationScope,
) -> bool:
    """Pure scope matcher — does not consume grant or execute mutations."""
    return (
        grant.mutation_id == scope.mutation_id
        and grant.mutation_type == scope.mutation_type
        and grant.tenant_id == scope.tenant_id
        and grant.resource_scope == scope.resource_scope
        and grant.resource_type == scope.resource_type
        and grant.resource_id == scope.resource_id
        and grant.current_revision == scope.current_revision
        and grant.target_revision == scope.target_revision
        and grant.task_id == scope.task_id
        and grant.run_id == scope.run_id
    )


def matches_request_to_grant(
    grant: ControlPlaneMutationApprovalGrant,
    request: ControlPlaneMutationRequest,
) -> bool:
    """Fail-closed request-to-grant binding for scoped approval consumption."""
    if request.approval_evidence_ref != grant.grant_id:
        return False
    if request.mutation_id != grant.mutation_id:
        return False
    if request.mutation_type != grant.mutation_type:
        return False
    if request.tenant_id != grant.tenant_id:
        return False
    if request.resource_scope != grant.resource_scope:
        return False
    if request.resource_type != grant.resource_type:
        return False
    if request.resource_id != grant.resource_id:
        return False
    if request.current_revision != grant.current_revision:
        return False
    if request.target_revision != grant.target_revision:
        return False
    request_digest = control_plane_mutation_request_digest(request)
    if request_digest != grant.request_digest:
        return False
    if str(request.task_id) if request.task_id is not None else None != grant.task_id:
        return False
    if str(request.run_id) if request.run_id is not None else None != grant.run_id:
        return False
    if request.principal.user_id != grant.service_principal_user_id:
        return False
    if request.principal.auth_subject != grant.service_principal_auth_subject:
        return False
    if request.principal.principal_type != grant.service_principal_type:
        return False
    return True


def _validate_human_approver(approver: RequestIdentity, tenant_id: str) -> None:
    if approver.principal_type is not PrincipalType.USER:
        raise ControlPlaneMutationApprovalError("human approver requires USER principal type")
    if not approver.user_id or not approver.auth_subject:
        raise ControlPlaneMutationApprovalError("human approver identity required")
    if approver.tenant_id != tenant_id:
        raise ControlPlaneMutationApprovalError("approver tenant_id mismatch")


@dataclass
class ControlPlaneMutationApprovalCoordinator:
    """In-process scoped approval store — not an authorization authority."""

    _grants: dict[str, ControlPlaneMutationApprovalGrant] = field(default_factory=dict)
    _consumed_grant_ids: set[str] = field(default_factory=set)
    _denials: dict[str, ControlPlaneMutationDenialRecord] = field(default_factory=dict)

    def get_grant(self, grant_id: str) -> ControlPlaneMutationApprovalGrant | None:
        if grant_id in self._consumed_grant_ids:
            return None
        return self._grants.get(grant_id)

    def is_consumed(self, grant_id: str) -> bool:
        return grant_id in self._consumed_grant_ids

    def get_denial(self, mutation_id: str) -> ControlPlaneMutationDenialRecord | None:
        return self._denials.get(mutation_id)

    def create_approval_grant(
        self,
        *,
        approver: RequestIdentity,
        service_principal: RequestIdentity,
        scope: ControlPlaneMutationAuthorizationScope,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence,
    ) -> ControlPlaneMutationApprovalGrant:
        _validate_human_approver(approver, scope.tenant_id)
        if authorization_evidence.mutation_id != scope.mutation_id:
            raise ControlPlaneMutationApprovalError("authorization evidence mutation_id mismatch")
        if authorization_evidence.request_digest.strip() == "":
            raise ControlPlaneMutationApprovalError("authorization evidence request_digest required")
        policy_rule_id = authorization_evidence.policy_rule_id.strip() or "ecp.control_plane_policy"
        grant_id = f"cpm-grant:{uuid4().hex}"
        grant = ControlPlaneMutationApprovalGrant(
            grant_id=grant_id,
            mutation_id=scope.mutation_id,
            mutation_type=scope.mutation_type,
            tenant_id=scope.tenant_id,
            resource_scope=scope.resource_scope,
            resource_type=scope.resource_type,
            resource_id=scope.resource_id,
            current_revision=scope.current_revision,
            target_revision=scope.target_revision,
            request_digest=authorization_evidence.request_digest,
            policy_rule_id=policy_rule_id,
            service_principal_user_id=service_principal.user_id or "",
            service_principal_auth_subject=service_principal.auth_subject or "",
            service_principal_type=service_principal.principal_type,
            approver_user_id=approver.user_id or "",
            approver_auth_subject=approver.auth_subject or "",
            approver_principal_type=approver.principal_type,
            approved_at=datetime.now(timezone.utc).isoformat(),
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
        self._grants[grant_id] = grant
        return grant

    def record_denial(
        self,
        *,
        approver: RequestIdentity,
        scope: ControlPlaneMutationAuthorizationScope,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence,
    ) -> ControlPlaneMutationDenialRecord:
        _validate_human_approver(approver, scope.tenant_id)
        policy_rule_id = authorization_evidence.policy_rule_id.strip() or "ecp.control_plane_policy"
        denial = ControlPlaneMutationDenialRecord(
            mutation_id=scope.mutation_id,
            mutation_type=scope.mutation_type,
            tenant_id=scope.tenant_id,
            resource_scope=scope.resource_scope,
            resource_type=scope.resource_type,
            resource_id=scope.resource_id,
            current_revision=scope.current_revision,
            target_revision=scope.target_revision,
            request_digest=authorization_evidence.request_digest,
            policy_rule_id=policy_rule_id,
            approver_user_id=approver.user_id or "",
            approver_auth_subject=approver.auth_subject or "",
            approver_principal_type=approver.principal_type,
            denied_at=datetime.now(timezone.utc).isoformat(),
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
        self._denials[scope.mutation_id] = denial
        return denial

    def consume_matching_grant(
        self,
        *,
        grant_id: str,
        request: ControlPlaneMutationRequest,
    ) -> ControlPlaneMutationApprovalGrant | None:
        """Consume grant before mutation execution — at-most-once approval authorization."""
        if grant_id in self._consumed_grant_ids:
            return None
        grant = self._grants.get(grant_id)
        if grant is None:
            return None
        if not matches_request_to_grant(grant, request):
            return None
        self._consumed_grant_ids.add(grant_id)
        del self._grants[grant_id]
        return grant


@dataclass(frozen=True, slots=True)
class ApprovalConsumingControlPlaneMutationEvaluator:
    """Wrap configured evaluator with scoped approval-evidence consumption."""

    inner: ControlPlaneMutationPolicyEvaluator
    coordinator: ControlPlaneMutationApprovalCoordinator

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        approval_ref = request.approval_evidence_ref
        if approval_ref is None:
            return self.inner.evaluate(request)
        consumed = self.coordinator.consume_matching_grant(
            grant_id=approval_ref,
            request=request,
        )
        if consumed is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="approval_evidence_invalid_or_consumed",
                policy_rule_id="control_plane_mutation.approval",
            )
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="scoped_human_approval_consumed",
            policy_rule_id=consumed.policy_rule_id,
            decision_id=f"cpm-allow:{consumed.grant_id}",
        )
