# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical MSE HITL effect gate after fresh Governance authorize (GR-10-R11).

Human judgment evidence and continuation grants never become Governance ALLOW.
This gate only decides whether a fresh authorization may proceed to physical effect:

* ALLOW → proceed (fresh permission)
* REQUIRE_HUMAN + exact matching grant → proceed once (grant consumed)
* REQUIRE_HUMAN without grant → require HITL (continuation request)
* ESCALATE → require HITL (never grant-executable)
* DENY / MODIFY / other → block
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    matches_current_requirement,
)
from intergrax.runtime.task.task import Task


class MseHitlEffectGateDisposition(Enum):
    """Disposition after fresh MSE authorization under HITL semantics."""

    PROCEED = auto()
    BLOCK = auto()
    REQUIRE_HITL = auto()


@dataclass(frozen=True, slots=True)
class MseHitlEffectGateOutcome:
    """Typed outcome of the MSE HITL effect gate."""

    disposition: MseHitlEffectGateDisposition
    authorization: MeaningfulSideEffectAuthorizationResult

    @property
    def governed_continuation_request(self) -> GovernedContinuationRequest | None:
        return self.authorization.governed_continuation_request


def evaluate_mse_hitl_effect_gate(
    authorization: MeaningfulSideEffectAuthorizationResult,
    *,
    enforcement_request: CollaborativeWorkEnforcementRequest,
    task: Task | None = None,
) -> MseHitlEffectGateOutcome:
    """Apply grant-matching HITL semantics to a fresh MSE authorization result.

    Does not execute side effects and does not mint grants. Pause materialization
    remains the caller's responsibility via governed continuation contracts.
    """
    action = authorization.decision.action
    enforcement = authorization.enforcement_result
    side_effect = enforcement_request.meaningful_side_effect_request
    resource_scope = enforcement_request.resource_scope or enforcement.authority_scope
    operation_id = enforcement.operation_id

    if action is PolicyAction.DENY or action is PolicyAction.MODIFY:
        if task is not None and side_effect is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )

    if action is PolicyAction.ALLOW and authorization.permitted:
        if task is not None and side_effect is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.PROCEED,
            authorization=authorization,
        )

    if action is PolicyAction.ESCALATE:
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.REQUIRE_HITL,
            authorization=authorization,
        )

    if action is PolicyAction.REQUIRE_HUMAN:
        if task is not None and side_effect is not None:
            stored_grant = task.runtime.governance.governed_continuation_grant
            if stored_grant is not None:
                if matches_current_requirement(
                    stored_grant,
                    current_side_effect=side_effect,
                    current_operation_id=operation_id,
                    current_resource_scope=resource_scope,
                    current_decision=authorization.decision,
                ):
                    consumed = GovernedContinuationGrantCoordinator.consume_matching_grant(
                        task,
                        expected_grant_id=stored_grant.grant_id,
                    )
                    if consumed is not None:
                        return MseHitlEffectGateOutcome(
                            disposition=MseHitlEffectGateDisposition.PROCEED,
                            authorization=authorization,
                        )
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.REQUIRE_HITL,
            authorization=authorization,
        )

    return MseHitlEffectGateOutcome(
        disposition=MseHitlEffectGateDisposition.BLOCK,
        authorization=authorization,
    )


__all__ = [
    "MseHitlEffectGateDisposition",
    "MseHitlEffectGateOutcome",
    "evaluate_mse_hitl_effect_gate",
]
