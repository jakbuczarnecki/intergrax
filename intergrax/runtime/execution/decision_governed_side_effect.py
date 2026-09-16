# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution Engine coordinator — Decision material → canonical Governance → effect (GR-6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.decision_authorization import (
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernanceMismatchError,
    DecisionGovernancePolicyContext,
)
from intergrax.contracts.decision_governance_material import (
    DecisionGovernanceMaterialMismatchError,
    decision_governance_material_ref_from_accepted,
    validate_decision_governance_material_for_authorization,
    validate_decision_governance_material_for_decision,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.runtime.decision_authorization import validate_execution_authorization_bundle
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_lifecycle import TaskLifecycle

T = TypeVar("T")
TResult = TypeVar("TResult")


class DecisionGovernedSideEffectError(RuntimeError):
    """Decision-bound side effect cannot proceed under canonical governance."""


def attach_decision_governance_material(
    side_effect: MeaningfulSideEffectRequest,
    *,
    decision: AuthoritativeAcceptedDecision[T],
    action: DecisionExecutionAction,
) -> MeaningfulSideEffectRequest:
    """Return side-effect request with typed immutable decision governance material."""
    material = decision_governance_material_ref_from_accepted(
        decision=decision,
        action=action,
    )
    return side_effect.model_copy(
        update={"decision_governance_material": material},
    )


def authorize_and_execute_decision_bound_side_effect(
    boundary: MeaningfulSideEffectAuthorizationBoundary,
    *,
    enforcement_request: CollaborativeWorkEnforcementRequest,
    decision: AuthoritativeAcceptedDecision[T],
    authorization: DecisionExecutionAuthorization,
    action: DecisionExecutionAction,
    policy_context: DecisionGovernancePolicyContext,
    execute: Callable[[], TResult],
    task: Task | None = None,
    lifecycle: TaskLifecycle | None = None,
    source_agent_id: str = "platform.decision_governed_side_effect",
    source_step_id: str | None = None,
    on_authorization: Callable[[MeaningfulSideEffectAuthorizationResult], None] | None = None,
) -> TResult | MeaningfulSideEffectAuthorizationResult:
    """Validate Decision provenance, then invoke canonical ``authorize_and_execute``."""
    side_effect = enforcement_request.meaningful_side_effect_request
    if side_effect is None:
        raise DecisionGovernedSideEffectError(
            "decision-bound side effect requires meaningful_side_effect_request",
        )
    try:
        validate_execution_authorization_bundle(
            authorization=authorization,
            decision=decision,
            action=action,
            current_policy_context=policy_context,
        )
        material = decision_governance_material_ref_from_accepted(
            decision=decision,
            action=action,
        )
        validate_decision_governance_material_for_decision(
            material=material,
            decision=decision,
            action=action,
        )
        validate_decision_governance_material_for_authorization(
            material=material,
            authorization=authorization,
            action=action,
        )
    except (DecisionGovernanceMaterialMismatchError, DecisionGovernanceMismatchError) as exc:
        raise DecisionGovernedSideEffectError(str(exc)) from exc

    bound_side_effect = attach_decision_governance_material(
        side_effect,
        decision=decision,
        action=action,
    )
    bound_request = enforcement_request.model_copy(
        update={"meaningful_side_effect_request": bound_side_effect},
    )
    return boundary.authorize_and_execute(
        bound_request,
        execute,
        task=task,
        lifecycle=lifecycle,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
        on_authorization=on_authorization,
    )


__all__ = [
    "DecisionGovernedSideEffectError",
    "attach_decision_governance_material",
    "authorize_and_execute_decision_bound_side_effect",
]
