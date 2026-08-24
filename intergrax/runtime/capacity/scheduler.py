# © Artur Czarnecki. All rights reserved.

"""Capacity evaluation scheduler (ECP-OBS.2)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
)
from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue, CapacityResumableMutation
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind, ScalingActionPlan
from intergrax.runtime.capacity.control_plane_governance import (
    EcpGovernanceBlockedError,
    EcpTenantScopeDenial,
    parse_celery_workers_revision,
    parse_k8s_replicas_revision,
)
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.events import PublishFn, publish_scale_applied, publish_scale_failed
from intergrax.runtime.capacity.governed_capacity_mutation import GovernedCapacityMutationExecutor
from intergrax.runtime.capacity.metrics import record_scale_action
from intergrax.runtime.capacity.provisioner import ScalingProvisioner, StaleCapacityStateError


TickFn = Callable[[], Awaitable[None]]


class SchedulerGovernanceBlockedError(Exception):
    """Automatic scheduler mutation blocked before provider side effect."""

    def __init__(self, blocker_code: str, message: str) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code


@dataclass(frozen=True, slots=True)
class SchedulerCapacityMutationBlocked:
    """Transient scheduler outcome when automatic capacity mutation is blocked."""

    action_id: str
    blocker_code: str
    policy_action: str | None = None
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence | None = None
    authorization_scope: ControlPlaneMutationAuthorizationScope | None = None
    tenant_scope_denial: EcpTenantScopeDenial | None = None


class CapacityScheduler:
    """Async cron driver that does not block Nexus."""

    def __init__(
        self,
        *,
        collector: CapacitySignalCollector,
        evaluator: ScalingEvaluator,
        provisioner: ScalingProvisioner | None = None,
        interval_seconds: float = 30.0,
        approval_queue: CapacityApprovalQueue | None = None,
        publish: PublishFn | None = None,
        execution_identity: RequestIdentity | None = None,
        governed_capacity_executor: GovernedCapacityMutationExecutor | None = None,
        tenant_id: str | None = None,
        k8s_deployment: str = "nexus-host",
        celery_pool_id: str = "default",
        requires_governed_execution: bool = False,
    ) -> None:
        self._collector = collector
        self._evaluator = evaluator
        self._provisioner = provisioner
        self._interval = interval_seconds
        self._approval_queue = approval_queue
        self._publish = publish
        self._execution_identity = execution_identity
        self._governed_capacity_executor = governed_capacity_executor
        self._tenant_id = tenant_id
        self._k8s_deployment = k8s_deployment
        self._celery_pool_id = celery_pool_id
        self._requires_governed_execution = requires_governed_execution
        self._task: asyncio.Task[None] | None = None
        self._blocked_outcomes: list[SchedulerCapacityMutationBlocked] = []

    def _require_provisioner(self) -> ScalingProvisioner:
        if self._provisioner is None:
            raise RuntimeError("capacity scheduler provisioner not configured")
        return self._provisioner

    async def _apply_plan(self, plan: ScalingActionPlan) -> None:
        for action in plan.actions:
            await self._apply_action(action)

    async def _apply_action(self, action: ScalingAction) -> None:
        if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
            if self._requires_governed_execution:
                self._apply_governed_k8s(action)
                return
            self._require_provisioner().apply(action, deployment=self._k8s_deployment)
            return
        if action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
            if self._requires_governed_execution:
                self._apply_governed_celery(action)
                return
            self._require_provisioner().apply(action)
            return
        if action.kind is ScalingActionKind.RAISE_ORCHESTRATION_CEILING:
            if self._requires_governed_execution:
                self._record_blocked_action(
                    action,
                    reason="orchestration_ceiling_governed_path_unavailable",
                    blocker_code="ECP_SCHEDULER_CEILING_UNSUPPORTED",
                )
                return
            self._require_provisioner().apply(action)
            return
        if action.kind is ScalingActionKind.REQUEST_HITL:
            return

    def _apply_governed_k8s(self, action: ScalingAction) -> None:
        try:
            principal = self._require_service_identity()
            tenant_id = self._require_tenant_id()
            executor = self._require_governed_executor()
        except SchedulerGovernanceBlockedError as exc:
            self._record_blocked_action(
                action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
            )
            return
        try:
            executor.scale_k8s_deployment(
                principal=principal,
                tenant_id=tenant_id,
                mutation_id=action.action_id,
                deployment=self._k8s_deployment,
                delta=action.delta,
            )
            self._record_applied_action(action)
        except EcpGovernanceBlockedError as exc:
            self._enqueue_blocked_mutation(action, exc)
            self._record_blocked_action(
                action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
                governance_error=exc,
            )
        except StaleCapacityStateError as exc:
            self._record_blocked_action(
                action,
                reason=str(exc),
                blocker_code="ECP_SCHEDULER_STALE_STATE",
                governance_error=exc,
            )

    def _apply_governed_celery(self, action: ScalingAction) -> None:
        try:
            principal = self._require_service_identity()
            tenant_id = self._require_tenant_id()
            executor = self._require_governed_executor()
        except SchedulerGovernanceBlockedError as exc:
            self._record_blocked_action(
                action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
            )
            return
        try:
            executor.scale_celery_workers(
                principal=principal,
                tenant_id=tenant_id,
                mutation_id=action.action_id,
                pool_id=self._celery_pool_id,
                delta=action.delta,
            )
            self._record_applied_action(action)
        except EcpGovernanceBlockedError as exc:
            self._enqueue_blocked_mutation(action, exc)
            self._record_blocked_action(
                action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
                governance_error=exc,
            )
        except StaleCapacityStateError as exc:
            self._record_blocked_action(
                action,
                reason=str(exc),
                blocker_code="ECP_SCHEDULER_STALE_STATE",
                governance_error=exc,
            )

    def _resume_governed_mutation(self, resumable: CapacityResumableMutation) -> None:
        try:
            principal = resumable.service_principal
            tenant_id = self._require_tenant_id()
            executor = self._require_governed_executor()
        except SchedulerGovernanceBlockedError as exc:
            self._record_blocked_action(
                resumable.action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
            )
            return
        action = resumable.action
        scope = resumable.authorization_scope
        if action.action_id != scope.mutation_id:
            self._record_blocked_action(
                action,
                reason="mutation_id mismatch",
                blocker_code="ECP_SCHEDULER_SCOPE_MISMATCH",
            )
            return
        try:
            if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
                executor.resume_k8s_deployment(
                    principal=principal,
                    tenant_id=tenant_id,
                    authorization_scope=scope,
                    approval_evidence_ref=resumable.approval_evidence_ref,
                )
            elif action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
                executor.resume_celery_workers(
                    principal=principal,
                    tenant_id=tenant_id,
                    authorization_scope=scope,
                    approval_evidence_ref=resumable.approval_evidence_ref,
                )
            else:
                self._record_blocked_action(
                    action,
                    reason="unsupported resume action kind",
                    blocker_code="ECP_SCHEDULER_RESUME_UNSUPPORTED",
                )
                return
            self._record_applied_action(action)
        except EcpGovernanceBlockedError as exc:
            self._record_blocked_action(
                action,
                reason=exc.blocker_code,
                blocker_code=exc.blocker_code,
                governance_error=exc,
            )
        except StaleCapacityStateError as exc:
            self._record_blocked_action(
                action,
                reason=str(exc),
                blocker_code="ECP_SCHEDULER_STALE_STATE",
                governance_error=exc,
            )

    def _resume_unrestricted_mutation(self, resumable: CapacityResumableMutation) -> None:
        action = resumable.action
        scope = resumable.authorization_scope
        coordinator = self._approval_queue.coordinator if self._approval_queue is not None else None
        if coordinator is None:
            self._record_blocked_action(
                action,
                reason="missing approval coordinator",
                blocker_code="ECP_SCHEDULER_MISSING_APPROVAL",
            )
            return
        grant = coordinator.get_grant(resumable.approval_evidence_ref)
        if grant is None:
            self._record_blocked_action(
                action,
                reason="approval grant missing or consumed",
                blocker_code="ECP_SCHEDULER_MISSING_APPROVAL",
            )
            return
        try:
            if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
                deployment, current_replicas = parse_k8s_replicas_revision(scope.current_revision)
                _, target_replicas = parse_k8s_replicas_revision(scope.target_revision)
                observed = self._require_provisioner().read_k8s_replicas(deployment=deployment)
                if observed != current_replicas:
                    raise StaleCapacityStateError(
                        authorized_current=current_replicas,
                        observed_current=observed,
                        deployment=deployment,
                    )
                from intergrax.runtime.capacity.control_plane_governance import (
                    build_scale_k8s_deployment_mutation_request,
                )

                mutation_request = build_scale_k8s_deployment_mutation_request(
                    principal=resumable.service_principal,
                    tenant_id=scope.tenant_id,
                    mutation_id=scope.mutation_id,
                    deployment=deployment,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    approval_evidence_ref=resumable.approval_evidence_ref,
                )
                consumed = coordinator.consume_matching_grant(
                    grant_id=resumable.approval_evidence_ref,
                    request=mutation_request,
                )
                if consumed is None:
                    self._record_blocked_action(
                        action,
                        reason="approval scope mismatch",
                        blocker_code="ECP_SCHEDULER_SCOPE_MISMATCH",
                    )
                    return
                self._require_provisioner()._apply_authorized_k8s_target(
                    deployment=deployment,
                    replicas=target_replicas,
                    authorized_current=current_replicas,
                )
            elif action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
                pool_id, current_workers = parse_celery_workers_revision(scope.current_revision)
                _, target_workers = parse_celery_workers_revision(scope.target_revision)
                observed = self._require_provisioner().read_celery_worker_count()
                if observed != current_workers:
                    raise StaleCapacityStateError(
                        authorized_current=current_workers,
                        observed_current=observed,
                        pool_id=pool_id,
                    )
                from intergrax.runtime.capacity.control_plane_governance import (
                    build_scale_celery_workers_mutation_request,
                )

                mutation_request = build_scale_celery_workers_mutation_request(
                    principal=resumable.service_principal,
                    tenant_id=scope.tenant_id,
                    mutation_id=scope.mutation_id,
                    pool_id=pool_id,
                    current_workers=current_workers,
                    target_workers=target_workers,
                    approval_evidence_ref=resumable.approval_evidence_ref,
                )
                consumed = coordinator.consume_matching_grant(
                    grant_id=resumable.approval_evidence_ref,
                    request=mutation_request,
                )
                if consumed is None:
                    self._record_blocked_action(
                        action,
                        reason="approval scope mismatch",
                        blocker_code="ECP_SCHEDULER_SCOPE_MISMATCH",
                    )
                    return
                self._require_provisioner()._apply_authorized_celery_target(
                    target_workers=target_workers,
                    authorized_current=current_workers,
                )
            else:
                self._record_blocked_action(
                    action,
                    reason="unsupported resume action kind",
                    blocker_code="ECP_SCHEDULER_RESUME_UNSUPPORTED",
                )
                return
            self._record_applied_action(action)
        except StaleCapacityStateError as exc:
            self._record_blocked_action(
                action,
                reason=str(exc),
                blocker_code="ECP_SCHEDULER_STALE_STATE",
                governance_error=exc,
            )

    def _enqueue_blocked_mutation(
        self,
        action: ScalingAction,
        exc: EcpGovernanceBlockedError,
    ) -> None:
        if self._approval_queue is None:
            return
        if exc.blocker_code != "ECP_BLOCKED_BY_REQUIRE_HUMAN":
            return
        if exc.authorization_scope is None or exc.authorization_evidence is None:
            return
        try:
            principal = self._require_service_identity()
        except SchedulerGovernanceBlockedError:
            return
        self._approval_queue.submit_pending(
            plan_id=f"blocked-{action.action_id}",
            action=action,
            authorization_scope=exc.authorization_scope,
            authorization_evidence=exc.authorization_evidence,
            service_principal=principal,
        )

    def _resolve_enqueue_principal(self) -> RequestIdentity | None:
        if self._execution_identity is not None:
            try:
                return self._require_service_identity()
            except SchedulerGovernanceBlockedError:
                return None
        if not self._requires_governed_execution:
            return RequestIdentity(
                tenant_id=self._tenant_id or "harness",
                user_id="capacity-harness",
                principal_type=PrincipalType.SERVICE,
                auth_subject="capacity-harness",
            )
        return None

    def _resolve_enqueue_tenant_id(self) -> str | None:
        if self._tenant_id is not None and self._tenant_id.strip():
            return self._tenant_id
        if not self._requires_governed_execution:
            return "harness"
        try:
            return self._require_tenant_id()
        except SchedulerGovernanceBlockedError:
            return None

    def _enqueue_hitl_plan(self, plan: ScalingActionPlan) -> None:
        if self._approval_queue is None:
            return
        principal = self._resolve_enqueue_principal()
        tenant_id = self._resolve_enqueue_tenant_id()
        if principal is None or tenant_id is None:
            return
        executor = self._governed_capacity_executor
        for action in plan.actions:
            if action.delta <= 0:
                continue
            if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
                if executor is not None:
                    try:
                        pending = executor.prepare_k8s_pending_authorization(
                            principal=principal,
                            tenant_id=tenant_id,
                            mutation_id=action.action_id,
                            deployment=self._k8s_deployment,
                            delta=action.delta,
                            translate_local_hitl=True,
                        )
                    except EcpGovernanceBlockedError:
                        continue
                    self._approval_queue.submit_pending(
                        plan_id=plan.plan_id,
                        action=action,
                        authorization_scope=pending.authorization_scope,
                        authorization_evidence=pending.authorization_evidence,
                        service_principal=principal,
                    )
                else:
                    self._enqueue_unrestricted_hitl(
                        plan_id=plan.plan_id,
                        action=action,
                        principal=principal,
                        tenant_id=tenant_id,
                    )
            elif action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
                if executor is not None:
                    try:
                        pending = executor.prepare_celery_pending_authorization(
                            principal=principal,
                            tenant_id=tenant_id,
                            mutation_id=action.action_id,
                            pool_id=self._celery_pool_id,
                            delta=action.delta,
                            translate_local_hitl=True,
                        )
                    except EcpGovernanceBlockedError:
                        continue
                    self._approval_queue.submit_pending(
                        plan_id=plan.plan_id,
                        action=action,
                        authorization_scope=pending.authorization_scope,
                        authorization_evidence=pending.authorization_evidence,
                        service_principal=principal,
                    )
                else:
                    self._enqueue_unrestricted_hitl(
                        plan_id=plan.plan_id,
                        action=action,
                        principal=principal,
                        tenant_id=tenant_id,
                    )

    def _enqueue_unrestricted_hitl(
        self,
        *,
        plan_id: str,
        action: ScalingAction,
        principal: RequestIdentity,
        tenant_id: str,
    ) -> None:
        from intergrax.contracts.control_plane_mutation import (
            authorization_scope_for_request,
            control_plane_mutation_request_digest,
            evidence_from_request_and_decision,
        )
        from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
        from intergrax.runtime.capacity.control_plane_governance import (
            build_scale_celery_workers_mutation_request,
            build_scale_k8s_deployment_mutation_request,
        )
        from intergrax.runtime.capacity.governed_capacity_mutation import LOCAL_HITL_POLICY_RULE_ID

        if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
            current = self._require_provisioner().read_k8s_replicas(deployment=self._k8s_deployment)
            target = max(0, current + action.delta)
            request = build_scale_k8s_deployment_mutation_request(
                principal=principal,
                tenant_id=tenant_id,
                mutation_id=action.action_id,
                deployment=self._k8s_deployment,
                current_replicas=current,
                target_replicas=target,
            )
        elif action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
            current = self._require_provisioner().read_celery_worker_count()
            target = max(1, current + action.delta)
            request = build_scale_celery_workers_mutation_request(
                principal=principal,
                tenant_id=tenant_id,
                mutation_id=action.action_id,
                pool_id=self._celery_pool_id,
                current_workers=current,
                target_workers=target,
            )
        else:
            return
        digest = control_plane_mutation_request_digest(request)
        decision = PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="ecp.local_hitl_for_scale_up",
            policy_rule_id=LOCAL_HITL_POLICY_RULE_ID,
        )
        evidence = evidence_from_request_and_decision(
            request,
            decision=decision,
            request_digest=digest,
        )
        scope = authorization_scope_for_request(request)
        self._approval_queue.submit_pending(
            plan_id=plan_id,
            action=action,
            authorization_scope=scope,
            authorization_evidence=evidence,
            service_principal=principal,
        )

    def _require_service_identity(self) -> RequestIdentity:
        if self._execution_identity is None:
            raise SchedulerGovernanceBlockedError(
                "ECP_SCHEDULER_MISSING_IDENTITY",
                "automatic capacity mutation requires configured SERVICE execution identity",
            )
        if self._execution_identity.principal_type is not PrincipalType.SERVICE:
            raise SchedulerGovernanceBlockedError(
                "ECP_SCHEDULER_INVALID_PRINCIPAL",
                "automatic capacity mutation requires SERVICE principal type",
            )
        return self._execution_identity

    def _require_tenant_id(self) -> str:
        if self._tenant_id is None or not self._tenant_id.strip():
            raise SchedulerGovernanceBlockedError(
                "ECP_SCHEDULER_MISSING_TENANT",
                "automatic capacity mutation requires configured tenant authority",
            )
        return self._tenant_id

    def _require_governed_executor(self) -> GovernedCapacityMutationExecutor:
        if self._governed_capacity_executor is None:
            raise SchedulerGovernanceBlockedError(
                "ECP_SCHEDULER_MISSING_EXECUTOR",
                "automatic capacity mutation requires governed capacity executor",
            )
        return self._governed_capacity_executor

    def _record_applied_action(self, action: ScalingAction) -> None:
        if self._requires_governed_execution and self._governed_capacity_executor is not None:
            self._governed_capacity_executor.record_scheduler_applied(
                action,
                tenant_id=self._tenant_id or "harness",
            )
            return
        provisioner = self._require_provisioner()
        provisioner.applied.append(action)
        record_scale_action(target=action.target.value)
        if self._publish is not None:
            publish_scale_applied(self._publish, action, tenant_id=self._tenant_id or "harness")

    @property
    def blocked_outcomes(self) -> tuple[SchedulerCapacityMutationBlocked, ...]:
        return tuple(self._blocked_outcomes)

    def _blocked_outcome_from_governance(
        self,
        action: ScalingAction,
        *,
        blocker_code: str,
        governance_error: EcpGovernanceBlockedError | StaleCapacityStateError | None,
    ) -> SchedulerCapacityMutationBlocked:
        if isinstance(governance_error, EcpGovernanceBlockedError):
            return SchedulerCapacityMutationBlocked(
                action_id=action.action_id,
                blocker_code=blocker_code,
                policy_action=governance_error.policy_action,
                authorization_evidence=governance_error.authorization_evidence,
                authorization_scope=governance_error.authorization_scope,
                tenant_scope_denial=governance_error.tenant_scope_denial,
            )
        return SchedulerCapacityMutationBlocked(
            action_id=action.action_id,
            blocker_code=blocker_code,
        )

    def _record_blocked_action(
        self,
        action: ScalingAction,
        *,
        reason: str,
        blocker_code: str,
        governance_error: EcpGovernanceBlockedError | StaleCapacityStateError | None = None,
    ) -> None:
        del reason
        blocked = self._blocked_outcome_from_governance(
            action,
            blocker_code=blocker_code,
            governance_error=governance_error,
        )
        self._blocked_outcomes.append(blocked)
        if self._governed_capacity_executor is not None:
            self._governed_capacity_executor.record_scheduler_blocked(
                action,
                blocker_code=blocker_code,
            )
        elif self._provisioner is not None:
            self._provisioner.failures.append(blocker_code)
        if self._publish is not None:
            publish_scale_failed(self._publish, action, reason=blocker_code)

    async def tick(self) -> None:
        if self._approval_queue is not None:
            for resumable in self._approval_queue.drain_resumable():
                if self._requires_governed_execution:
                    self._resume_governed_mutation(resumable)
                else:
                    self._resume_unrestricted_mutation(resumable)

        signals = self._collector.collect()
        plan = self._evaluator.evaluate(signals)
        if plan.evaluation_status == "hitl_required":
            self._enqueue_hitl_plan(plan)
            if self._publish is not None:
                from intergrax.runtime.capacity.events import publish_scale_requested

                publish_scale_requested(self._publish, plan)
            return
        if plan.evaluation_status != "planned":
            return
        await self._apply_plan(plan)

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
