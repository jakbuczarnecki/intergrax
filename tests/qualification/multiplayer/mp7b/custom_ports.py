# © Artur Czarnecki. All rights reserved.

"""Typed conforming test doubles for MeaningfulSideEffectAuthorizationPort (MP-7B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision


def _result(
    request: CollaborativeWorkEnforcementRequest,
    *,
    permitted: bool,
    action: PolicyAction,
    reason: str,
) -> MeaningfulSideEffectAuthorizationResult:
    decision = PolicyDecision(
        action=action, reason=reason, policy_rule_id="mp7b.custom.port"
    )
    return MeaningfulSideEffectAuthorizationResult(
        permitted=permitted,
        decision=decision,
        enforcement_result=CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        ),
        requires_governed_continuation=False,
    )


@dataclass(frozen=True, slots=True)
class AllowingAuthorizationPort:
    """Custom conforming implementation that always permits."""

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        del source_agent_id, source_step_id
        return _result(
            request,
            permitted=True,
            action=PolicyAction.ALLOW,
            reason="mp7b-custom-allow",
        )


@dataclass(frozen=True, slots=True)
class DenyingAuthorizationPort:
    """Custom conforming implementation that always denies."""

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        del source_agent_id, source_step_id
        return _result(
            request,
            permitted=False,
            action=PolicyAction.DENY,
            reason="mp7b-custom-deny",
        )


@dataclass(frozen=True, slots=True)
class ExplodingAuthorizationPort:
    """Custom conforming implementation that raises (consumer must fail closed)."""

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        del request, source_agent_id, source_step_id
        raise RuntimeError("mp7b-custom-port-exploded")


__all__ = [
    "AllowingAuthorizationPort",
    "DenyingAuthorizationPort",
    "ExplodingAuthorizationPort",
]
