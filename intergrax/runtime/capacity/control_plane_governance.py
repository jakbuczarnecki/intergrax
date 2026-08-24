# © Artur Czarnecki. All rights reserved.

"""ECP control-plane mutation helpers (CLA-CPM-ECP)."""

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
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

K8S_DEPLOYMENT_RESOURCE_TYPE = "ecp.kubernetes_deployment"
CELERY_POOL_RESOURCE_TYPE = "ecp.celery_pool"
MUTATION_TYPE_SCALE_K8S_DEPLOYMENT = "ecp.scale_k8s_deployment"
MUTATION_TYPE_SCALE_CELERY_WORKERS = "ecp.scale_celery_workers"


class EcpResourceTenantResolver(Protocol):
    """Resolve tenant authority scope for ECP capacity resources."""

    def resolve_k8s_deployment_tenant(self, *, deployment: str) -> str:
        """Return canonical tenant id owning ``deployment``."""

    def resolve_celery_pool_tenant(self, *, pool_id: str) -> str:
        """Return canonical tenant id owning ``pool_id``."""


@dataclass(frozen=True, slots=True)
class StaticEcpResourceTenantResolver:
    """Explicit single-tenant host mapping for ECP control-plane mutation authority."""

    tenant_id: str

    def resolve_k8s_deployment_tenant(self, *, deployment: str) -> str:
        del deployment
        return self.tenant_id

    def resolve_celery_pool_tenant(self, *, pool_id: str) -> str:
        del pool_id
        return self.tenant_id


class EcpTenantScopeDenial(BaseModel):
    """Pre-evaluation tenant authority rejection — no mutation request was evaluated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    resource_type: str
    resource_id: str
    resource_scope: str
    principal_type: PrincipalType
    principal_user_id: str | None = None
    principal_auth_subject: str | None = None
    reason: str


class EcpGovernanceBlockedError(Exception):
    """ECP control-plane mutation blocked before or during governance."""

    def __init__(
        self,
        blocker_code: str,
        message: str,
        *,
        policy_action: str,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None,
        authorization_scope: ControlPlaneMutationAuthorizationScope | None = None,
        tenant_scope_denial: EcpTenantScopeDenial | None = None,
    ) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code
        self.policy_action = policy_action
        self.authorization_evidence = authorization_evidence
        self.authorization_scope = authorization_scope
        self.tenant_scope_denial = tenant_scope_denial


def k8s_deployment_resource_scope(*, tenant_id: str, deployment: str) -> str:
    return f"ecp.tenant:{tenant_id}.deployment:{deployment}"


def celery_pool_resource_scope(*, tenant_id: str, pool_id: str) -> str:
    return f"ecp.tenant:{tenant_id}.celery_pool:{pool_id}"


def k8s_replicas_revision_token(*, deployment: str, replicas: int) -> str:
    return f"k8s_deployment:{deployment}:replicas:{replicas}"


def celery_workers_revision_token(*, pool_id: str, worker_count: int) -> str:
    return f"celery_pool:{pool_id}:workers:{worker_count}"


def parse_k8s_replicas_revision(revision: str) -> tuple[str, int]:
    prefix = "k8s_deployment:"
    suffix = ":replicas:"
    if not revision.startswith(prefix):
        raise ValueError("invalid k8s replicas revision")
    body = revision[len(prefix):]
    deployment, separator, count_text = body.partition(suffix)
    if not separator or not deployment or not count_text:
        raise ValueError("invalid k8s replicas revision")
    return deployment, int(count_text)


def parse_celery_workers_revision(revision: str) -> tuple[str, int]:
    prefix = "celery_pool:"
    suffix = ":workers:"
    if not revision.startswith(prefix):
        raise ValueError("invalid celery workers revision")
    body = revision[len(prefix):]
    pool_id, separator, count_text = body.partition(suffix)
    if not separator or not pool_id or not count_text:
        raise ValueError("invalid celery workers revision")
    return pool_id, int(count_text)


def build_scale_k8s_deployment_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    mutation_id: str,
    deployment: str,
    current_replicas: int,
    target_replicas: int,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
        principal=principal,
        resource_scope=k8s_deployment_resource_scope(tenant_id=tenant_id, deployment=deployment),
        resource_type=K8S_DEPLOYMENT_RESOURCE_TYPE,
        resource_id=deployment,
        current_revision=k8s_replicas_revision_token(
            deployment=deployment,
            replicas=current_replicas,
        ),
        target_revision=k8s_replicas_revision_token(
            deployment=deployment,
            replicas=target_replicas,
        ),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
        approval_evidence_ref=approval_evidence_ref,
    )


def build_scale_celery_workers_mutation_request(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    mutation_id: str,
    pool_id: str,
    current_workers: int,
    target_workers: int,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=MUTATION_TYPE_SCALE_CELERY_WORKERS,
        principal=principal,
        resource_scope=celery_pool_resource_scope(tenant_id=tenant_id, pool_id=pool_id),
        resource_type=CELERY_POOL_RESOURCE_TYPE,
        resource_id=pool_id,
        current_revision=celery_workers_revision_token(
            pool_id=pool_id,
            worker_count=current_workers,
        ),
        target_revision=celery_workers_revision_token(
            pool_id=pool_id,
            worker_count=target_workers,
        ),
        risk_classification=ControlPlaneMutationRisk.MEDIUM,
        approval_evidence_ref=approval_evidence_ref,
    )


def _parse_ecp_resource_scope(resource_scope: str) -> tuple[str, str, str]:
    tenant_prefix = "ecp.tenant:"
    if not resource_scope.startswith(tenant_prefix):
        raise ValueError("invalid ecp resource_scope")
    remainder = resource_scope[len(tenant_prefix):]
    if ".deployment:" in remainder:
        tenant_id, separator, resource_name = remainder.partition(".deployment:")
        resource_kind = "deployment"
    elif ".celery_pool:" in remainder:
        tenant_id, separator, resource_name = remainder.partition(".celery_pool:")
        resource_kind = "celery_pool"
    else:
        raise ValueError("invalid ecp resource_scope")
    if not separator or not tenant_id or not resource_name:
        raise ValueError("invalid ecp resource_scope")
    return tenant_id, resource_kind, resource_name


class EcpTenantScopedControlPlaneMutationEvaluator:
    """Fail closed when tenant authority or permission policy is not explicitly configured."""

    def __init__(
        self,
        *,
        tenant_resolver: EcpResourceTenantResolver | None = None,
        inner: ControlPlaneMutationPolicyEvaluator | None = None,
    ) -> None:
        self._tenant_resolver = tenant_resolver
        self._inner = inner

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        if self._tenant_resolver is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_not_configured",
                policy_rule_id="ecp.tenant_scope",
            )
        tenant_id, resource_kind, resource_name = _parse_ecp_resource_scope(request.resource_scope)
        if resource_kind == "deployment":
            environment_tenant = self._tenant_resolver.resolve_k8s_deployment_tenant(
                deployment=resource_name,
            )
        elif resource_kind == "celery_pool":
            environment_tenant = self._tenant_resolver.resolve_celery_pool_tenant(
                pool_id=resource_name,
            )
        else:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="unsupported_ecp_resource_scope",
                policy_rule_id="ecp.tenant_scope",
            )
        if environment_tenant != request.principal.tenant_id:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_mismatch",
                policy_rule_id="ecp.tenant_scope",
            )
        if environment_tenant != tenant_id:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="tenant_authority_mismatch",
                policy_rule_id="ecp.tenant_scope",
            )
        if self._inner is None:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="control_plane_policy_not_configured",
                policy_rule_id="ecp.control_plane_policy",
            )
        return self._inner.evaluate(request)


def compose_ecp_tenant_scoped_mutation_boundary(
    *,
    policy_evaluator: ControlPlaneMutationPolicyEvaluator,
    tenant_resolver: EcpResourceTenantResolver,
) -> ControlPlaneMutationAuthorizationBoundary:
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=EcpTenantScopedControlPlaneMutationEvaluator(
            tenant_resolver=tenant_resolver,
            inner=policy_evaluator,
        ),
    )


def authorize_scoped_ecp_control_plane_mutation(
    *,
    boundary: ControlPlaneMutationAuthorizationBoundary,
    tenant_resolver: EcpResourceTenantResolver,
    request: ControlPlaneMutationRequest,
) -> ControlPlaneMutationAuthorizationResult:
    scoped_boundary = compose_ecp_tenant_scoped_mutation_boundary(
        policy_evaluator=boundary.evaluator,
        tenant_resolver=tenant_resolver,
    )
    return scoped_boundary.authorize(request)


def validate_ecp_principal_tenant_authority(
    *,
    principal: RequestIdentity,
    tenant_id: str,
    resource_type: str,
    resource_id: str,
    resource_scope: str,
    operation: str,
) -> None:
    if principal.tenant_id != tenant_id:
        raise EcpGovernanceBlockedError(
            "ECP_BLOCKED_BY_TENANT_AUTHORITY",
            f"{operation} denied by tenant authority scope",
            policy_action=PolicyAction.DENY.value,
            tenant_scope_denial=EcpTenantScopeDenial(
                tenant_id=tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_scope=resource_scope,
                principal_type=principal.principal_type,
                principal_user_id=principal.user_id,
                principal_auth_subject=principal.auth_subject,
                reason="principal_tenant_mismatch",
            ),
        )


def validate_ecp_resource_tenant_authority(
    *,
    tenant_id: str,
    tenant_resolver: EcpResourceTenantResolver,
    resource_type: str,
    resource_id: str,
    resource_scope: str,
    principal: RequestIdentity,
    operation: str,
    deployment: str | None = None,
    pool_id: str | None = None,
) -> None:
    validate_ecp_principal_tenant_authority(
        principal=principal,
        tenant_id=tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_scope=resource_scope,
        operation=operation,
    )
    if deployment is not None:
        resolved = tenant_resolver.resolve_k8s_deployment_tenant(deployment=deployment)
        if resolved != tenant_id:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_BY_TENANT_AUTHORITY",
                f"{operation} denied by tenant authority scope",
                policy_action=PolicyAction.DENY.value,
                tenant_scope_denial=EcpTenantScopeDenial(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_scope=resource_scope,
                    principal_type=principal.principal_type,
                    principal_user_id=principal.user_id,
                    principal_auth_subject=principal.auth_subject,
                    reason="resource_tenant_mismatch",
                ),
            )
    if pool_id is not None:
        resolved = tenant_resolver.resolve_celery_pool_tenant(pool_id=pool_id)
        if resolved != tenant_id:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_BY_TENANT_AUTHORITY",
                f"{operation} denied by tenant authority scope",
                policy_action=PolicyAction.DENY.value,
                tenant_scope_denial=EcpTenantScopeDenial(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    resource_scope=resource_scope,
                    principal_type=principal.principal_type,
                    principal_user_id=principal.user_id,
                    principal_auth_subject=principal.auth_subject,
                    reason="resource_tenant_mismatch",
                ),
            )


def enforce_ecp_authorization_result(
    result: ControlPlaneMutationAuthorizationResult,
    *,
    operation: str,
) -> ControlPlaneMutationAuthorizationResult:
    if result.permitted:
        return result
    action = result.decision.action
    if result.decision.reason == "tenant_authority_mismatch":
        raise EcpGovernanceBlockedError(
            "ECP_BLOCKED_BY_TENANT_AUTHORITY",
            f"{operation} denied by tenant authority scope",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.REQUIRE_HUMAN:
        raise EcpGovernanceBlockedError(
            "ECP_BLOCKED_BY_REQUIRE_HUMAN",
            f"{operation} requires governed human approval",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    if action is PolicyAction.ESCALATE:
        raise EcpGovernanceBlockedError(
            "ECP_BLOCKED_BY_ESCALATE",
            f"{operation} requires escalation",
            policy_action=action.value,
            authorization_evidence=result.evidence,
            authorization_scope=result.authorization_scope,
        )
    raise EcpGovernanceBlockedError(
        "ECP_BLOCKED_BY_POLICY",
        f"{operation} denied by control-plane policy",
        policy_action=action.value,
        authorization_evidence=result.evidence,
        authorization_scope=result.authorization_scope,
    )
