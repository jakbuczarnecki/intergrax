# © Artur Czarnecki. All rights reserved.

"""Capacity action governance (ECP-7.2 / ECP-PROD.6)."""

from __future__ import annotations

from intergrax.runtime.capacity.contracts import ScalingActionPlan


def requires_hitl_approval(plan: ScalingActionPlan) -> bool:
    """True when scale-up must wait for operator approval."""
    return plan.evaluation_status == "hitl_required"


def approved_plan_for_provision(plan: ScalingActionPlan) -> ScalingActionPlan:
    """Convert a HITL-gated plan into a provisionable plan after operator approval."""
    if plan.evaluation_status != "hitl_required":
        return plan
    return plan.model_copy(update={"evaluation_status": "planned"})
