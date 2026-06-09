# © Artur Czarnecki. All rights reserved.

"""Capacity action governance (ECP-7.2)."""

from __future__ import annotations

from intergrax.runtime.capacity.contracts import ScalingActionPlan


def requires_hitl_approval(plan: ScalingActionPlan) -> bool:
    """True when scale-up must wait for operator approval."""
    return plan.evaluation_status == "hitl_required"
