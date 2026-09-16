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
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
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
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
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

    def __init__(
        self,
        *,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        inner_execution_guard: CanonicalInnerExecutionGuardPort,
    ) -> None:
        self._enforcement_gate = enforcement_gate
        self._inner_execution_guard = inner_execution_guard

    @staticmethod
    def _inner_enforcement_denied(
        request: CollaborativeWorkEnforcementRequest,
        *,
        reason: str,
    ) -> MeaningfulSideEffectAuthorizationResult:
        deny = PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            policy_rule_id="platform.canonical_inner_enforcement",
        )
        composition = PolicyCompositionResult(
            decision=deny,
            collaborative_authority=deny,
        )
        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=composition,
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=False,
            decision=deny,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )

    def _assert_inner_execution(
        self,
        request: CollaborativeWorkEnforcementRequest,
    ) -> MeaningfulSideEffectAuthorizationResult | None:
        side_effect = request.meaningful_side_effect_request
        if side_effect is None:
            return self._inner_enforcement_denied(
                request,
                reason="meaningful side effect request required for inner enforcement",
            )
        try:
            self._inner_execution_guard.assert_meaningful_side_effect_bound(side_effect)
        except CanonicalInnerGovernanceViolation as exc:
            return self._inner_enforcement_denied(request, reason=exc.reason)
        except RuntimeError as exc:
            return self._inner_enforcement_denied(request, reason=str(exc))
        return None

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        inner_block = self._assert_inner_execution(request)
        if inner_block is not None:
            return inner_block
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
        on_authorization: Callable[[MeaningfulSideEffectAuthorizationResult], None] | None = None,
        hitl_continuation: InternalOrchestrationContinuation | None = None,
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
        if on_authorization is not None:
            on_authorization(authorization)
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
                    hitl_continuation=hitl_continuation,
                )
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
            return authorization

        return authorization
