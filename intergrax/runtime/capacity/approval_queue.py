# © Artur Czarnecki. All rights reserved.

"""Capacity scale-up approval queue — pending work holder, not authorization authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationApprovalGrant,
    ControlPlaneMutationDenialRecord,
)
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionPlan
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ControlPlaneMutationApprovalCoordinator,
    ControlPlaneMutationApprovalError,
)

ApprovalStatus = Literal["pending", "approved", "denied", "consumed"]


@dataclass(frozen=True, slots=True)
class CapacityApprovalRecord:
    """One pending or resolved capacity mutation awaiting or after human decision."""

    mutation_id: str
    plan_id: str
    action: ScalingAction
    authorization_scope: ControlPlaneMutationAuthorizationScope
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence
    service_principal: RequestIdentity
    status: ApprovalStatus
    requested_at: datetime
    approval_grant_id: str | None = None
    denial_record: ControlPlaneMutationDenialRecord | None = None


@dataclass(frozen=True, slots=True)
class CapacityResumableMutation:
    """Approved mutation ready for governed resume with scoped approval evidence."""

    mutation_id: str
    plan_id: str
    action: ScalingAction
    authorization_scope: ControlPlaneMutationAuthorizationScope
    approval_evidence_ref: str
    service_principal: RequestIdentity


@dataclass
class CapacityApprovalQueue:
    """In-process continuation holder for HITL-gated capacity mutations."""

    coordinator: ControlPlaneMutationApprovalCoordinator = field(
        default_factory=ControlPlaneMutationApprovalCoordinator,
    )
    _records: dict[str, CapacityApprovalRecord] = field(default_factory=dict)
    _approved_mutation_ids: list[str] = field(default_factory=list)

    def submit_pending(
        self,
        *,
        plan_id: str,
        action: ScalingAction,
        authorization_scope: ControlPlaneMutationAuthorizationScope,
        authorization_evidence: ControlPlaneMutationAuthorizationEvidence,
        service_principal: RequestIdentity,
    ) -> str:
        if authorization_scope.mutation_id != action.action_id:
            msg = "authorization scope mutation_id must match action.action_id"
            raise ValueError(msg)
        if self._records.get(action.action_id) is not None:
            existing = self._records[action.action_id]
            if existing.status in ("pending", "approved"):
                return action.action_id
        self._records[action.action_id] = CapacityApprovalRecord(
            mutation_id=action.action_id,
            plan_id=plan_id,
            action=action,
            authorization_scope=authorization_scope,
            authorization_evidence=authorization_evidence,
            service_principal=service_principal,
            status="pending",
            requested_at=datetime.now(timezone.utc),
        )
        return action.action_id

    def submit(self, plan: ScalingActionPlan) -> str:
        """Legacy plan submit — requires pre-built pending records via submit_pending."""
        if plan.evaluation_status != "hitl_required":
            msg = f"plan {plan.plan_id} is not hitl_required"
            raise ValueError(msg)
        if not plan.actions:
            msg = f"plan {plan.plan_id} has no actions for hitl submit"
            raise ValueError(msg)
        first = plan.actions[0]
        if self._records.get(first.action_id) is None:
            msg = (
                f"plan {plan.plan_id} requires canonical pending mutations via submit_pending"
            )
            raise ValueError(msg)
        return plan.plan_id

    def approve_mutation(
        self,
        mutation_id: str,
        approver: RequestIdentity,
    ) -> ControlPlaneMutationApprovalGrant | None:
        record = self._records.get(mutation_id)
        if record is None or record.status != "pending":
            return None
        try:
            grant = self.coordinator.create_approval_grant(
                approver=approver,
                service_principal=record.service_principal,
                scope=record.authorization_scope,
                authorization_evidence=record.authorization_evidence,
            )
        except ControlPlaneMutationApprovalError:
            return None
        self._records[mutation_id] = CapacityApprovalRecord(
            mutation_id=record.mutation_id,
            plan_id=record.plan_id,
            action=record.action,
            authorization_scope=record.authorization_scope,
            authorization_evidence=record.authorization_evidence,
            service_principal=record.service_principal,
            status="approved",
            requested_at=record.requested_at,
            approval_grant_id=grant.grant_id,
        )
        self._approved_mutation_ids.append(mutation_id)
        return grant

    def approve(
        self,
        plan_id: str,
        approver: RequestIdentity | None = None,
    ) -> ScalingActionPlan | None:
        """Plan-level approve without approver cannot authorize mutations."""
        if approver is None:
            return None
        pending_for_plan = [
            record for record in self._records.values()
            if record.plan_id == plan_id and record.status == "pending"
        ]
        if not pending_for_plan:
            return None
        approved_actions: list[ScalingAction] = []
        for record in pending_for_plan:
            grant = self.approve_mutation(record.mutation_id, approver)
            if grant is None:
                return None
            approved_actions.append(record.action)
        return ScalingActionPlan(
            plan_id=plan_id,
            actions=tuple(approved_actions),
            evaluation_status="planned",
        )

    def deny_mutation(
        self,
        mutation_id: str,
        approver: RequestIdentity,
    ) -> ControlPlaneMutationDenialRecord | None:
        record = self._records.get(mutation_id)
        if record is None or record.status != "pending":
            return None
        try:
            denial = self.coordinator.record_denial(
                approver=approver,
                scope=record.authorization_scope,
                authorization_evidence=record.authorization_evidence,
            )
        except ControlPlaneMutationApprovalError:
            return None
        self._records[mutation_id] = CapacityApprovalRecord(
            mutation_id=record.mutation_id,
            plan_id=record.plan_id,
            action=record.action,
            authorization_scope=record.authorization_scope,
            authorization_evidence=record.authorization_evidence,
            service_principal=record.service_principal,
            status="denied",
            requested_at=record.requested_at,
            denial_record=denial,
        )
        return denial

    def deny(self, plan_id: str, approver: RequestIdentity | None = None) -> bool:
        if approver is None:
            return False
        pending_for_plan = [
            record for record in self._records.values()
            if record.plan_id == plan_id and record.status == "pending"
        ]
        if not pending_for_plan:
            return False
        for record in pending_for_plan:
            if self.deny_mutation(record.mutation_id, approver) is None:
                return False
        return True

    def list_pending(self) -> list[CapacityApprovalRecord]:
        return [record for record in self._records.values() if record.status == "pending"]

    def drain_resumable(self) -> list[CapacityResumableMutation]:
        resumable: list[CapacityResumableMutation] = []
        for mutation_id in self._approved_mutation_ids:
            record = self._records.get(mutation_id)
            if (
                record is not None
                and record.status == "approved"
                and record.approval_grant_id is not None
            ):
                resumable.append(
                    CapacityResumableMutation(
                        mutation_id=record.mutation_id,
                        plan_id=record.plan_id,
                        action=record.action,
                        authorization_scope=record.authorization_scope,
                        approval_evidence_ref=record.approval_grant_id,
                        service_principal=record.service_principal,
                    ),
                )
                self._records[mutation_id] = CapacityApprovalRecord(
                    mutation_id=record.mutation_id,
                    plan_id=record.plan_id,
                    action=record.action,
                    authorization_scope=record.authorization_scope,
                    authorization_evidence=record.authorization_evidence,
                    service_principal=record.service_principal,
                    status="consumed",
                    requested_at=record.requested_at,
                    approval_grant_id=record.approval_grant_id,
                )
        self._approved_mutation_ids.clear()
        return resumable

    def drain_approved(self) -> list[ScalingActionPlan]:
        """Legacy drain — groups resumable mutations by plan for unrestricted harness paths."""
        plans_by_id: dict[str, list[ScalingAction]] = {}
        for item in self.drain_resumable():
            plans_by_id.setdefault(item.plan_id, []).append(item.action)
        return [
            ScalingActionPlan(
                plan_id=plan_id,
                actions=tuple(actions),
                evaluation_status="planned",
            )
            for plan_id, actions in plans_by_id.items()
        ]

    def scope_matches_pending(
        self,
        mutation_id: str,
        scope: ControlPlaneMutationAuthorizationScope,
    ) -> bool:
        record = self._records.get(mutation_id)
        if record is None:
            return False
        pending_scope = record.authorization_scope
        return (
            pending_scope.mutation_id == scope.mutation_id
            and pending_scope.mutation_type == scope.mutation_type
            and pending_scope.tenant_id == scope.tenant_id
            and pending_scope.resource_scope == scope.resource_scope
            and pending_scope.resource_type == scope.resource_type
            and pending_scope.resource_id == scope.resource_id
            and pending_scope.current_revision == scope.current_revision
            and pending_scope.target_revision == scope.target_revision
            and pending_scope.task_id == scope.task_id
            and pending_scope.run_id == scope.run_id
        )
