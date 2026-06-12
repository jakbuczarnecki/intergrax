# © Artur Czarnecki. All rights reserved.

"""Capacity action governance (ECP-7.2 / ECP-PROD.6)."""

from __future__ import annotations

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


def approve_capacity_plan(
    queue: CapacityApprovalQueue,
    plan_id: str,
    *,
    publish: PublishFn | None = None,
) -> ScalingActionPlan | None:
    """Operator approves a pending capacity plan."""
    approved = queue.approve(plan_id)
    if approved is not None and publish is not None:
        publish_scale_approved(publish, approved)
    return approved


def deny_capacity_plan(
    queue: CapacityApprovalQueue,
    plan_id: str,
    *,
    publish: PublishFn | None = None,
) -> bool:
    """Operator denies a pending capacity plan."""
    denied = queue.deny(plan_id)
    if denied and publish is not None:
        publish_scale_denied(publish, plan_id)
    return denied
