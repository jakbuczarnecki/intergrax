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
    compose_governed_continuation_from_enforcement,
)

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
    ) -> T | MeaningfulSideEffectAuthorizationResult:
        """Evaluate enforcement before invoking ``execute`` — never calls it unless ALLOW."""
        authorization = self.authorize(request)
        if not authorization.permitted:
            return authorization
        return execute()
