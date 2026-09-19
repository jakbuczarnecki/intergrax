# © Artur Czarnecki. All rights reserved.

"""Canonical meaningful side-effect authorization contract (ADR-GR-10-002 / GR-10-R9-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision


@dataclass(frozen=True, slots=True)
class MeaningfulSideEffectAuthorizationResult:
    """Outcome of collaborative-work enforcement at the shared side-effect boundary."""

    permitted: bool
    decision: PolicyDecision
    enforcement_result: CollaborativeWorkEnforcementResult
    requires_governed_continuation: bool
    governed_continuation_request: GovernedContinuationRequest | None = None


class MeaningfulSideEffectAuthorizationConsistencyError(ValueError):
    """Raised when a port returns semantically inconsistent authorization data."""


def assert_consistent_meaningful_side_effect_authorization_result(
    result: MeaningfulSideEffectAuthorizationResult,
) -> None:
    """Fail closed when ``permitted`` disagrees with ``decision.action``."""
    action = result.decision.action
    if result.permitted and action is not PolicyAction.ALLOW:
        raise MeaningfulSideEffectAuthorizationConsistencyError(
            "permitted=True requires decision.action ALLOW",
        )
    if not result.permitted and action is PolicyAction.ALLOW:
        raise MeaningfulSideEffectAuthorizationConsistencyError(
            "permitted=False incompatible with decision.action ALLOW",
        )


@runtime_checkable
class MeaningfulSideEffectAuthorizationPort(Protocol):
    """Authorize consequential tool effects before physical execution."""

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        """Evaluate enforcement for a proposed meaningful side effect."""


__all__ = [
    "MeaningfulSideEffectAuthorizationConsistencyError",
    "MeaningfulSideEffectAuthorizationPort",
    "MeaningfulSideEffectAuthorizationResult",
    "assert_consistent_meaningful_side_effect_authorization_result",
]
