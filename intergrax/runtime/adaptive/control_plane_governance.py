# © Artur Czarnecki. All rights reserved.

"""AHI control-plane mutation helpers for profile apply/rollback (CLA-CPM-AHI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.adaptive.contracts import ProfileArtifactType
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

AHI_PROFILE_POINTER_RESOURCE_TYPE = "ahi.profile_pointer"
MUTATION_TYPE_APPLY_PROFILE = "ahi.apply_profile"
MUTATION_TYPE_ROLLBACK_PROFILE = "ahi.rollback_profile"

_ACTIVE_NONE_TOKEN = "__none__"


class AhiTenantScopeResolver(Protocol):
    """Resolve canonical tenant authority for one tenant/task_class scope."""

    def resolve_tenant_id(self, *, tenant_id: str, task_class: str) -> str:
        """Return canonical tenant id owning ``tenant_id`` / ``task_class``."""


@dataclass(frozen=True, slots=True)
class DirectAhiTenantScopeResolver:
    """Explicit tenant mapping for AHI control-plane mutation authority."""

    def resolve_tenant_id(self, *, tenant_id: str, task_class: str) -> str:
        del task_class
        return tenant_id


class AhiTenantScopeDenial(BaseModel):
    """Pre-evaluation tenant authority rejection — no mutation request was evaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_class: str
    resource_type: str
    resource_id: str
    resource_scope: str
    principal_type: PrincipalType
    principal_user_id: str | None = None
    principal_auth_subject: str | None = None
    reason: str


class AhiGovernanceBlockedError(Exception):
    """AHI control-plane mutation blocked by governance before domain commit."""

    def __init__(
        self,
        blocker_code: str,
        message: str,
        *,
        policy_action: str,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None,
        authorization_scope: ControlPlaneMutationAuthorizationScope | None = None,
        tenant_scope_denial: AhiTenantScopeDenial | None = None,
    ) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code
        self.policy_action = policy_action
        self.authorization_evidence = authorization_evidence
        self.authorization_scope = authorization_scope
        self.tenant_scope_denial = tenant_scope_denial


def ahi_resource_scope(*, tenant_id: str, task_class: str) -> str:
    return f"ahi.tenant:{tenant_id}.task_class:{task_class}"


def ahi_resource_id(artifact_type: ProfileArtifactType) -> str:
    return artifact_type.value


def profile_pointer_revision_token(
    artifact_type: ProfileArtifactType,
    active_version_id: str | None,
) -> str:
    version_token = active_version_id if active_version_id is not None else _ACTIVE_NONE_TOKEN
    return f"profile_pointer:{artifact_type.value}|active:{version_token}"


def build_apply_profile_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_class: str,
    mutation_id: str,
    artifact_type: ProfileArtifactType,
    current_active_version_id: str | None,
    target_version_id: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_APPLY_PROFILE,
        principal=principal,
        resource_scope=ahi_resource_scope(tenant_id=tenant_id, task_class=task_class),
        resource_type=AHI_PROFILE_POINTER_RESOURCE_TYPE,
        resource_id=ahi_resource_id(artifact_type),
        current_revision=profile_pointer_revision_token(
            artifact_type,
            current_active_version_id,
        ),
        target_revision=profile_pointer_revision_token(
            artifact_type,
            target_version_id,
        ),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
    )


def build_rollback_profile_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_class: str,
    mutation_id: str,
    artifact_type: ProfileArtifactType,
    current_active_version_id: str,
    target_previous_version_id: str,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_ROLLBACK_PROFILE,
        principal=principal,
        resource_scope=ahi_resource_scope(tenant_id=tenant_id, task_class=task_class),
        resource_type=AHI_PROFILE_POINTER_RESOURCE_TYPE,
        resource_id=ahi_resource_id(artifact_type),
        current_revision=profile_pointer_revision_token(
            artifact_type,
            current_active_version_id,
        ),
        target_revision=profile_pointer_revision_token(
            artifact_type,
            target_previous_version_id,
        ),
        risk_classification=ControlPlaneMutationRisk.HIGH,
    )


def _parse_ahi_resource_scope(resource_scope: str) -> tuple[str, str]:
    tenant_prefix = "ahi.tenant:"
    task_prefix = ".task_class:"
    if not resource_scope.startswith(tenant_prefix):
        raise ValueError("invalid ahi resource_scope")
    tenant_part, separator, task_part = resource_scope[len(tenant_prefix):].partition(task_prefix)
    if not separator or not tenant_part or not task_part:
        raise ValueError("invalid ahi resource_scope")
    return tenant_part, task_part


class AhiTenantScopedControlPlaneMutationEvaluator:
    """Fail closed when tenant authority or permission policy is not explicitly configured."""

    def __init__(
        self,
        *,
        tenant_resolver: AhiTenantScopeResolver | None = None,
        inner: ControlPlaneMutationPolicyEvaluator | None = None,
    ) -> None:
        self._tenant_resolver = tenant_resolver
        self._inner = inner

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        if self._tenant_resolver is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_not_configured",
                policy_rule_id="ahi.tenant_scope",
            )
        tenant_id, task_class = _parse_ahi_resource_scope(request.resource_scope)
        environment_tenant = self._tenant_resolver.resolve_tenant_id(
            tenant_id=tenant_id,
            task_class=task_class,
        )
        if environment_tenant != request.principal.tenant_id:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_mismatch",
                policy_rule_id="ahi.tenant_scope",
            )
        if self._inner is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="control_plane_policy_not_configured",
                policy_rule_id="ahi.control_plane_policy",
            )
        return self._inner.evaluate(request)


def compose_ahi_tenant_scoped_mutation_boundary(
    *,
    policy_evaluator: ControlPlaneMutationPolicyEvaluator,
    tenant_resolver: AhiTenantScopeResolver,
) -> ControlPlaneMutationAuthorizationBoundary:
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=AhiTenantScopedControlPlaneMutationEvaluator(
            tenant_resolver=tenant_resolver,
            inner=policy_evaluator,
        )
    )


def authorize_scoped_ahi_control_plane_mutation(
    *,
    boundary: ControlPlaneMutationAuthorizationBoundary,
    tenant_resolver: AhiTenantScopeResolver,
    request: ControlPlaneMutationRequest,
) -> ControlPlaneMutationAuthorizationResult:
    scoped_boundary = compose_ahi_tenant_scoped_mutation_boundary(
        policy_evaluator=boundary.evaluator,
        tenant_resolver=tenant_resolver,
    )
    return scoped_boundary.authorize(request)


def validate_ahi_principal_tenant_authority(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    task_class: str,
    artifact_type: ProfileArtifactType,
    operation: str,
) -> None:
    if principal.tenant_id != tenant_id:
        raise AhiGovernanceBlockedError(
            "AHI_BLOCKED_BY_TENANT_AUTHORITY",
            f"{operation} denied by tenant authority scope",
            policy_action=PolicyAction.DENY.value,
            tenant_scope_denial=AhiTenantScopeDenial(
                tenant_id=tenant_id,
                task_class=task_class,
                resource_type=AHI_PROFILE_POINTER_RESOURCE_TYPE,
                resource_id=ahi_resource_id(artifact_type),
                resource_scope=ahi_resource_scope(tenant_id=tenant_id, task_class=task_class),
                principal_type=principal.principal_type,
                principal_user_id=principal.user_id,
                principal_auth_subject=principal.auth_subject,
                reason="principal_tenant_mismatch",
            ),
        )


def enforce_ahi_authorization_result(
    result: ControlPlaneMutationAuthorizationResult,
    *,
    operation: str,
) -> ControlPlaneMutationAuthorizationResult:
    if result.permitted:
        return result
    action = result.decision.action
    if result.decision.reason == "tenant_authority_mismatch":
        raise AhiGovernanceBlockedError(
            "AHI_BLOCKED_BY_TENANT_AUTHORITY",
            f"{operation} denied by tenant authority scope",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.REQUIRE_HUMAN:
        raise AhiGovernanceBlockedError(
            "AHI_BLOCKED_BY_REQUIRE_HUMAN",
            f"{operation} requires governed human approval",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.ESCALATE:
        raise AhiGovernanceBlockedError(
            "AHI_BLOCKED_BY_ESCALATE",
            f"{operation} requires escalation",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    raise AhiGovernanceBlockedError(
        "AHI_BLOCKED_BY_POLICY",
        f"{operation} denied by control-plane policy",
        policy_action=action.value,
        authorization_evidence=result.evidence,
        authorization_scope=result.authorization_scope,
    )
