# © Artur Czarnecki. All rights reserved.

"""Canonical pre-side-effect authorization boundary (COLLAB-WORK-1H).

Invokes ``CollaborativeWorkEnforcementGate`` immediately before a proposed
meaningful side effect may proceed. Evaluation only — execution remains owned
by the caller/runtime layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    matches_current_requirement,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_lifecycle import TaskLifecycle, TaskState

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MeaningfulSideEffectAuthorizationResult:
    """Outcome of collaborative-work enforcement at the shared side-effect boundary."""

    permitted: bool
    decision: PolicyDecision
    enforcement_result: CollaborativeWorkEnforcementResult
    requires_governed_continuation: bool
    governed_continuation_request: GovernedContinuationRequest | None = None


class MeaningfulSideEffectAuthorizationBoundary:
    """Shared production boundary for collaborative enforcement before side effects."""

    def __init__(self, *, enforcement_gate: CollaborativeWorkEnforcementGate) -> None:
        self._enforcement_gate = enforcement_gate

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        enforcement_result = self._enforcement_gate.evaluate(request)
        decision = enforcement_result.composition.decision
        action = decision.action
        permitted = action is PolicyAction.ALLOW
        requires_continuation = action in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE)
        governed_continuation_request = compose_governed_continuation_from_enforcement(
            request,
            decision=decision,
            enforcement_operation_id=enforcement_result.operation_id,
            enforcement_authority_scope=enforcement_result.authority_scope,
            requires_governed_continuation=requires_continuation,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=permitted,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=requires_continuation,
            governed_continuation_request=governed_continuation_request,
        )

    def authorize_and_execute(
        self,
        request: CollaborativeWorkEnforcementRequest,
        execute: Callable[[], T],
        *,
        task: Task | None = None,
        lifecycle: TaskLifecycle | None = None,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
    ) -> T | MeaningfulSideEffectAuthorizationResult:
        """Fresh enforcement evaluation before ``execute``.

        ``PolicyAction.ALLOW`` executes without a grant. Fresh ``DENY`` is absolute —
        approval grants cannot override it. Fresh ``REQUIRE_HUMAN`` executes only when a
        stored grant exactly matches the current requirement and side-effect identity;
        the grant is consumed before ``execute`` (at-most-once approval authorization,
        not exactly-once external execution). Without a matching grant, canonical HITL
        pause is entered when ``lifecycle`` is supplied.
        """
        authorization = self.authorize(
            request,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        action = authorization.decision.action
        enforcement = authorization.enforcement_result
        side_effect = request.meaningful_side_effect_request
        resource_scope = request.resource_scope or enforcement.authority_scope
        operation_id = enforcement.operation_id

        if action is PolicyAction.DENY:
            if task is not None and side_effect is not None:
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
            return authorization

        if action is PolicyAction.ALLOW:
            if task is not None and side_effect is not None:
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
            return execute()

        if action is PolicyAction.REQUIRE_HUMAN:
            if task is None or side_effect is None:
                return authorization

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
                        return execute()
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )

            if (
                lifecycle is not None
                and authorization.governed_continuation_request is not None
            ):
                apply_governed_continuation_pause(
                    task,
                    authorization.governed_continuation_request,
                )
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
            return authorization

        return authorization
