# © Artur Czarnecki. All rights reserved.

"""AHI proposal → capacity ceiling bridge (ECP-8.1)."""

from __future__ import annotations

from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind, ScalingTarget


def scaling_action_from_ahi_proposal(*, ceiling_delta: int, reason: str) -> ScalingAction:
    """Map approved adaptive proposal to orchestration ceiling raise."""
    return ScalingAction(
        kind=ScalingActionKind.RAISE_ORCHESTRATION_CEILING,
        target=ScalingTarget.ORCHESTRATION_CEILING,
        delta=ceiling_delta,
        reason=reason,
    )
