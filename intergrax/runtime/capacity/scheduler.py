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
from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind, ScalingActionPlan
from intergrax.runtime.capacity.control_plane_governance import (
    EcpGovernanceBlockedError,
    EcpTenantScopeDenial,
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
        provisioner: ScalingProvisioner,
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

    async def _apply_plan(self, plan: ScalingActionPlan) -> None:
        for action in plan.actions:
            await self._apply_action(action)

    async def _apply_action(self, action: ScalingAction) -> None:
        if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
            if self._requires_governed_execution:
                self._apply_governed_k8s(action)
                return
            self._provisioner.apply(action, deployment=self._k8s_deployment)
            return
        if action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
            if self._requires_governed_execution:
                self._apply_governed_celery(action)
                return
            self._provisioner.apply(action)
            return
        if action.kind is ScalingActionKind.RAISE_ORCHESTRATION_CEILING:
            if self._requires_governed_execution:
                self._record_blocked_action(
                    action,
                    reason="orchestration_ceiling_governed_path_unavailable",
                    blocker_code="ECP_SCHEDULER_CEILING_UNSUPPORTED",
                )
                return
            self._provisioner.apply(action)
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
        self._provisioner.applied.append(action)
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
        self._provisioner.failures.append(blocker_code)
        if self._publish is not None:
            publish_scale_failed(self._publish, action, reason=blocker_code)

    async def tick(self) -> None:
        if self._approval_queue is not None:
            for approved_plan in self._approval_queue.drain_approved():
                await self._apply_plan(approved_plan)

        signals = self._collector.collect()
        plan = self._evaluator.evaluate(signals)
        if plan.evaluation_status == "hitl_required":
            if self._approval_queue is not None:
                self._approval_queue.submit(plan)
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
