# © Artur Czarnecki. All rights reserved.

"""Governed ECP capacity mutation facade (CLA-CPM-ECP)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationRequest,
    authorization_scope_for_request,
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.capacity.control_plane_governance import (
    CELERY_POOL_RESOURCE_TYPE,
    EcpGovernanceBlockedError,
    EcpResourceTenantResolver,
    K8S_DEPLOYMENT_RESOURCE_TYPE,
    authorize_scoped_ecp_control_plane_mutation,
    build_scale_celery_workers_mutation_request,
    build_scale_k8s_deployment_mutation_request,
    celery_pool_resource_scope,
    enforce_ecp_authorization_result,
    k8s_deployment_resource_scope,
    parse_celery_workers_revision,
    parse_k8s_replicas_revision,
    validate_ecp_resource_tenant_authority,
)
from intergrax.runtime.capacity.provisioner import ScalingProvisioner, StaleCapacityStateError
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
    ControlPlaneMutationApprovalCoordinator,
)

LOCAL_HITL_POLICY_RULE_ID = "ecp.require_hitl_for_scale_up"


@dataclass(frozen=True, slots=True)
class GovernedCapacityMutationSuccess:
    """Successful governed capacity mutation with canonical authorization evidence."""

    authorization_evidence: ControlPlaneMutationAuthorizationEvidence
    authorization_result: ControlPlaneMutationAuthorizationResult
    applied_target_revision: str


@dataclass(frozen=True, slots=True)
class GovernedCapacityPendingAuthorization:
    """Canonical pending authorization scope for HITL continuation."""

    request: ControlPlaneMutationRequest
    authorization_scope: ControlPlaneMutationAuthorizationScope
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence
    authorization_result: ControlPlaneMutationAuthorizationResult


class GovernedCapacityMutationExecutor:
    """Canonical control-plane governed facade for ECP K8s/Celery capacity mutations."""

    def __init__(
        self,
        *,
        provisioner: ScalingProvisioner,
        mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
        tenant_resolver: EcpResourceTenantResolver | None,
        approval_coordinator: ControlPlaneMutationApprovalCoordinator | None = None,
    ) -> None:
        self._provisioner = provisioner
        self._tenant_resolver = tenant_resolver
        self._approval_coordinator = approval_coordinator
        self._mutation_boundary = self._wrap_boundary(mutation_boundary, approval_coordinator)

    @staticmethod
    def _wrap_boundary(
        mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
        approval_coordinator: ControlPlaneMutationApprovalCoordinator | None,
    ) -> ControlPlaneMutationAuthorizationBoundary | None:
        if mutation_boundary is None:
            return None
        if approval_coordinator is None:
            return mutation_boundary
        wrapped_evaluator = ApprovalConsumingControlPlaneMutationEvaluator(
            inner=mutation_boundary.evaluator,
            coordinator=approval_coordinator,
        )
        return ControlPlaneMutationAuthorizationBoundary(evaluator=wrapped_evaluator)

    def prepare_k8s_pending_authorization(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        mutation_id: str,
        deployment: str,
        delta: int,
        translate_local_hitl: bool = False,
    ) -> GovernedCapacityPendingAuthorization:
        operation = "prepare_k8s_pending_authorization"
        boundary = self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        resource_scope = k8s_deployment_resource_scope(tenant_id=tenant_id, deployment=deployment)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=K8S_DEPLOYMENT_RESOURCE_TYPE,
            resource_id=deployment,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            deployment=deployment,
        )
        current_replicas = self._provisioner.read_k8s_replicas(deployment=deployment)
        target_replicas = max(0, current_replicas + delta)
        request = build_scale_k8s_deployment_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=mutation_id,
            deployment=deployment,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
        )
        return self._finalize_pending_authorization(
            request=request,
            boundary=boundary,
            tenant_resolver=tenant_resolver,
            translate_local_hitl=translate_local_hitl,
        )

    def prepare_celery_pending_authorization(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        mutation_id: str,
        pool_id: str,
        delta: int,
        translate_local_hitl: bool = False,
    ) -> GovernedCapacityPendingAuthorization:
        operation = "prepare_celery_pending_authorization"
        boundary = self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        resource_scope = celery_pool_resource_scope(tenant_id=tenant_id, pool_id=pool_id)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=CELERY_POOL_RESOURCE_TYPE,
            resource_id=pool_id,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            pool_id=pool_id,
        )
        current_workers = self._provisioner.read_celery_worker_count()
        target_workers = max(1, current_workers + delta)
        request = build_scale_celery_workers_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=mutation_id,
            pool_id=pool_id,
            current_workers=current_workers,
            target_workers=target_workers,
        )
        return self._finalize_pending_authorization(
            request=request,
            boundary=boundary,
            tenant_resolver=tenant_resolver,
            translate_local_hitl=translate_local_hitl,
        )

    def _finalize_pending_authorization(
        self,
        *,
        request: ControlPlaneMutationRequest,
        boundary: ControlPlaneMutationAuthorizationBoundary,
        tenant_resolver: EcpResourceTenantResolver,
        translate_local_hitl: bool,
    ) -> GovernedCapacityPendingAuthorization:
        authorization_result = authorize_scoped_ecp_control_plane_mutation(
            boundary=boundary,
            tenant_resolver=tenant_resolver,
            request=request,
        )
        if authorization_result.validation_failed:
            enforce_ecp_authorization_result(authorization_result, operation="pending_authorization")
        if authorization_result.decision.action is PolicyAction.DENY:
            enforce_ecp_authorization_result(authorization_result, operation="pending_authorization")
        evidence = authorization_result.evidence
        if translate_local_hitl and authorization_result.decision.action is PolicyAction.ALLOW:
            digest = control_plane_mutation_request_digest(request)
            translated = PolicyDecision(
                action=PolicyAction.REQUIRE_HUMAN,
                reason="ecp.local_hitl_for_scale_up",
                policy_rule_id=LOCAL_HITL_POLICY_RULE_ID,
            )
            evidence = evidence_from_request_and_decision(
                request,
                decision=translated,
                request_digest=digest,
            )
        scope = authorization_scope_for_request(request)
        return GovernedCapacityPendingAuthorization(
            request=request,
            authorization_scope=scope,
            authorization_evidence=evidence,
            authorization_result=authorization_result,
        )

    def scale_k8s_deployment(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        mutation_id: str,
        deployment: str,
        delta: int,
        approval_evidence_ref: str | None = None,
    ) -> GovernedCapacityMutationSuccess:
        operation = "scale_k8s_deployment"
        self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        resource_scope = k8s_deployment_resource_scope(tenant_id=tenant_id, deployment=deployment)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=K8S_DEPLOYMENT_RESOURCE_TYPE,
            resource_id=deployment,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            deployment=deployment,
        )
        current_replicas = self._provisioner.read_k8s_replicas(deployment=deployment)
        target_replicas = max(0, current_replicas + delta)
        request = build_scale_k8s_deployment_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=mutation_id,
            deployment=deployment,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            approval_evidence_ref=approval_evidence_ref,
        )
        return self._authorize_and_apply_k8s(
            request=request,
            tenant_resolver=tenant_resolver,
            operation=operation,
            deployment=deployment,
            authorized_current=current_replicas,
            target_replicas=target_replicas,
        )

    def resume_k8s_deployment(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        authorization_scope: ControlPlaneMutationAuthorizationScope,
        approval_evidence_ref: str,
    ) -> GovernedCapacityMutationSuccess:
        operation = "resume_k8s_deployment"
        self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        deployment, current_replicas = parse_k8s_replicas_revision(
            authorization_scope.current_revision,
        )
        _, target_replicas = parse_k8s_replicas_revision(authorization_scope.target_revision)
        if deployment != authorization_scope.resource_id:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_SCOPE_MISMATCH",
                f"{operation} denied by authorization scope mismatch",
                policy_action=PolicyAction.DENY.value,
            )
        resource_scope = k8s_deployment_resource_scope(tenant_id=tenant_id, deployment=deployment)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=K8S_DEPLOYMENT_RESOURCE_TYPE,
            resource_id=deployment,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            deployment=deployment,
        )
        request = build_scale_k8s_deployment_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=authorization_scope.mutation_id,
            deployment=deployment,
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            approval_evidence_ref=approval_evidence_ref,
        )
        return self._authorize_and_apply_k8s(
            request=request,
            tenant_resolver=tenant_resolver,
            operation=operation,
            deployment=deployment,
            authorized_current=current_replicas,
            target_replicas=target_replicas,
        )

    def _authorize_and_apply_k8s(
        self,
        *,
        request: ControlPlaneMutationRequest,
        tenant_resolver: EcpResourceTenantResolver,
        operation: str,
        deployment: str,
        authorized_current: int,
        target_replicas: int,
    ) -> GovernedCapacityMutationSuccess:
        authorization_result = authorize_scoped_ecp_control_plane_mutation(
            boundary=self._mutation_boundary,
            tenant_resolver=tenant_resolver,
            request=request,
        )
        enforce_ecp_authorization_result(authorization_result, operation=operation)
        observed_current = self._provisioner.read_k8s_replicas(deployment=deployment)
        if observed_current != authorized_current:
            raise StaleCapacityStateError(
                authorized_current=authorized_current,
                observed_current=observed_current,
                deployment=deployment,
            )
        self._provisioner._apply_authorized_k8s_target(
            deployment=deployment,
            replicas=target_replicas,
            authorized_current=authorized_current,
        )
        return GovernedCapacityMutationSuccess(
            authorization_evidence=authorization_result.evidence,
            authorization_result=authorization_result,
            applied_target_revision=request.target_revision,
        )

    def scale_celery_workers(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        mutation_id: str,
        pool_id: str,
        delta: int,
        approval_evidence_ref: str | None = None,
    ) -> GovernedCapacityMutationSuccess:
        operation = "scale_celery_workers"
        self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        resource_scope = celery_pool_resource_scope(tenant_id=tenant_id, pool_id=pool_id)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=CELERY_POOL_RESOURCE_TYPE,
            resource_id=pool_id,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            pool_id=pool_id,
        )
        current_workers = self._provisioner.read_celery_worker_count()
        target_workers = max(1, current_workers + delta)
        request = build_scale_celery_workers_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=mutation_id,
            pool_id=pool_id,
            current_workers=current_workers,
            target_workers=target_workers,
            approval_evidence_ref=approval_evidence_ref,
        )
        return self._authorize_and_apply_celery(
            request=request,
            tenant_resolver=tenant_resolver,
            operation=operation,
            pool_id=pool_id,
            authorized_current=current_workers,
            target_workers=target_workers,
        )

    def resume_celery_workers(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        authorization_scope: ControlPlaneMutationAuthorizationScope,
        approval_evidence_ref: str,
    ) -> GovernedCapacityMutationSuccess:
        operation = "resume_celery_workers"
        self._require_boundary(operation)
        tenant_resolver = self._require_tenant_resolver(operation)
        pool_id, current_workers = parse_celery_workers_revision(
            authorization_scope.current_revision,
        )
        _, target_workers = parse_celery_workers_revision(authorization_scope.target_revision)
        if pool_id != authorization_scope.resource_id:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_SCOPE_MISMATCH",
                f"{operation} denied by authorization scope mismatch",
                policy_action=PolicyAction.DENY.value,
            )
        resource_scope = celery_pool_resource_scope(tenant_id=tenant_id, pool_id=pool_id)
        validate_ecp_resource_tenant_authority(
            tenant_id=tenant_id,
            tenant_resolver=tenant_resolver,
            resource_type=CELERY_POOL_RESOURCE_TYPE,
            resource_id=pool_id,
            resource_scope=resource_scope,
            principal=principal,
            operation=operation,
            pool_id=pool_id,
        )
        request = build_scale_celery_workers_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            mutation_id=authorization_scope.mutation_id,
            pool_id=pool_id,
            current_workers=current_workers,
            target_workers=target_workers,
            approval_evidence_ref=approval_evidence_ref,
        )
        return self._authorize_and_apply_celery(
            request=request,
            tenant_resolver=tenant_resolver,
            operation=operation,
            pool_id=pool_id,
            authorized_current=current_workers,
            target_workers=target_workers,
        )

    def _authorize_and_apply_celery(
        self,
        *,
        request: ControlPlaneMutationRequest,
        tenant_resolver: EcpResourceTenantResolver,
        operation: str,
        pool_id: str,
        authorized_current: int,
        target_workers: int,
    ) -> GovernedCapacityMutationSuccess:
        authorization_result = authorize_scoped_ecp_control_plane_mutation(
            boundary=self._mutation_boundary,
            tenant_resolver=tenant_resolver,
            request=request,
        )
        enforce_ecp_authorization_result(authorization_result, operation=operation)
        observed_current = self._provisioner.read_celery_worker_count()
        if observed_current != authorized_current:
            raise StaleCapacityStateError(
                authorized_current=authorized_current,
                observed_current=observed_current,
                pool_id=pool_id,
            )
        self._provisioner._apply_authorized_celery_target(
            target_workers=target_workers,
            authorized_current=authorized_current,
        )
        return GovernedCapacityMutationSuccess(
            authorization_evidence=authorization_result.evidence,
            authorization_result=authorization_result,
            applied_target_revision=request.target_revision,
        )

    def _require_boundary(self, operation: str) -> ControlPlaneMutationAuthorizationBoundary:
        if self._mutation_boundary is None:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_MISSING_BOUNDARY",
                f"{operation} requires configured control-plane mutation boundary",
                policy_action=PolicyAction.DENY.value,
            )
        return self._mutation_boundary

    def _require_tenant_resolver(self, operation: str) -> EcpResourceTenantResolver:
        if self._tenant_resolver is None:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_MISSING_TENANT_RESOLVER",
                f"{operation} requires configured tenant authority resolver",
                policy_action=PolicyAction.DENY.value,
            )
        return self._tenant_resolver
