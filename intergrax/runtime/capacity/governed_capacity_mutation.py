# © Artur Czarnecki. All rights reserved.

"""Governed ECP capacity mutation facade (CLA-CPM-ECP)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction
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
    validate_ecp_resource_tenant_authority,
)
from intergrax.runtime.capacity.provisioner import ScalingProvisioner, StaleCapacityStateError
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


@dataclass(frozen=True, slots=True)
class GovernedCapacityMutationSuccess:
    """Successful governed capacity mutation with canonical authorization evidence."""

    authorization_evidence: ControlPlaneMutationAuthorizationEvidence
    authorization_result: ControlPlaneMutationAuthorizationResult
    applied_target_revision: str


class GovernedCapacityMutationExecutor:
    """Canonical control-plane governed facade for ECP K8s/Celery capacity mutations."""

    def __init__(
        self,
        *,
        provisioner: ScalingProvisioner,
        mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
        tenant_resolver: EcpResourceTenantResolver | None,
    ) -> None:
        self._provisioner = provisioner
        self._mutation_boundary = mutation_boundary
        self._tenant_resolver = tenant_resolver

    def scale_k8s_deployment(
        self,
        *,
        principal: RequestIdentity,
        tenant_id: str,
        mutation_id: str,
        deployment: str,
        delta: int,
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
        )
        authorization_result = authorize_scoped_ecp_control_plane_mutation(
            boundary=self._mutation_boundary,
            tenant_resolver=tenant_resolver,
            request=request,
        )
        enforce_ecp_authorization_result(authorization_result, operation=operation)
        observed_current = self._provisioner.read_k8s_replicas(deployment=deployment)
        if observed_current != current_replicas:
            raise StaleCapacityStateError(
                authorized_current=current_replicas,
                observed_current=observed_current,
                deployment=deployment,
            )
        self._provisioner._apply_authorized_k8s_target(
            deployment=deployment,
            replicas=target_replicas,
            authorized_current=current_replicas,
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
        )
        authorization_result = authorize_scoped_ecp_control_plane_mutation(
            boundary=self._mutation_boundary,
            tenant_resolver=tenant_resolver,
            request=request,
        )
        enforce_ecp_authorization_result(authorization_result, operation=operation)
        observed_current = self._provisioner.read_celery_worker_count()
        if observed_current != current_workers:
            raise StaleCapacityStateError(
                authorized_current=current_workers,
                observed_current=observed_current,
                pool_id=pool_id,
            )
        self._provisioner._apply_authorized_celery_target(
            target_workers=target_workers,
            authorized_current=current_workers,
        )
        return GovernedCapacityMutationSuccess(
            authorization_evidence=authorization_result.evidence,
            authorization_result=authorization_result,
            applied_target_revision=request.target_revision,
        )

    def _require_boundary(self, operation: str) -> None:
        if self._mutation_boundary is None:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_MISSING_BOUNDARY",
                f"{operation} requires configured control-plane mutation boundary",
                policy_action=PolicyAction.DENY.value,
            )

    def _require_tenant_resolver(self, operation: str) -> EcpResourceTenantResolver:
        if self._tenant_resolver is None:
            raise EcpGovernanceBlockedError(
                "ECP_BLOCKED_MISSING_TENANT_RESOLVER",
                f"{operation} requires configured tenant authority resolver",
                policy_action=PolicyAction.DENY.value,
            )
        return self._tenant_resolver
