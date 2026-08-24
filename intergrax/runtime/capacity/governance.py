# © Artur Czarnecki. All rights reserved.

"""Capacity action governance (ECP-7.2 / ECP-PROD.6)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationApprovalGrant
from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue
from intergrax.runtime.capacity.contracts import ScalingActionPlan
from intergrax.runtime.capacity.events import PublishFn, publish_scale_approved, publish_scale_denied


def requires_hitl_approval(plan: ScalingActionPlan) -> bool:
    """True when scale-up must wait for operator approval."""
    return plan.evaluation_status == "hitl_required"


def approved_plan_for_provision(plan: ScalingActionPlan) -> ScalingActionPlan:
    """Convert a HITL-gated plan into a provisionable plan after operator approval."""
    if plan.evaluation_status != "hitl_required":
        return plan
    return plan.model_copy(update={"evaluation_status": "planned"})


def approve_capacity_mutation(
    queue: CapacityApprovalQueue,
    mutation_id: str,
    approver: RequestIdentity,
    *,
    publish: PublishFn | None = None,
) -> ControlPlaneMutationApprovalGrant | None:
    """Operator approves one exact pending capacity mutation."""
    grant = queue.approve_mutation(mutation_id, approver)
    if grant is not None and publish is not None:
        record = queue._records.get(mutation_id)
        if record is not None:
            publish_scale_approved(
                publish,
                ScalingActionPlan(
                    plan_id=record.plan_id,
                    actions=(record.action,),
                    evaluation_status="planned",
                ),
            )
    return grant


def approve_capacity_plan(
    queue: CapacityApprovalQueue,
    plan_id: str,
    approver: RequestIdentity,
    *,
    publish: PublishFn | None = None,
) -> ScalingActionPlan | None:
    """Operator approves all pending mutations in a plan — each action scoped separately."""
    approved = queue.approve(plan_id, approver)
    if approved is not None and publish is not None:
        publish_scale_approved(publish, approved)
    return approved


def deny_capacity_mutation(
    queue: CapacityApprovalQueue,
    mutation_id: str,
    approver: RequestIdentity,
    *,
    publish: PublishFn | None = None,
) -> bool:
    """Operator denies one exact pending capacity mutation."""
    denied = queue.deny_mutation(mutation_id, approver) is not None
    if denied and publish is not None:
        publish_scale_denied(publish, mutation_id)
    return denied


def deny_capacity_plan(
    queue: CapacityApprovalQueue,
    plan_id: str,
    approver: RequestIdentity,
    *,
    publish: PublishFn | None = None,
) -> bool:
    """Operator denies all pending mutations in a plan."""
    denied = queue.deny(plan_id, approver)
    if denied and publish is not None:
        publish_scale_denied(publish, plan_id)
    return denied
