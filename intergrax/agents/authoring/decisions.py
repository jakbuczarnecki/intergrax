# © Artur Czarnecki. All rights reserved.

"""UAEP decision helpers for authored agents (Phase DX-2.4)."""

from __future__ import annotations

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType


def complete(*, reason: str = "step finished") -> AgentDecision:
    return AgentDecision(type=AgentDecisionType.COMPLETE, reason=reason)


def continue_to(step_id: str, *, reason: str = "continue") -> AgentDecision:
    return AgentDecision(
        type=AgentDecisionType.CONTINUE,
        next_step_id=step_id,
        reason=reason,
    )


def delegate_to(agent_id: str, *, reason: str = "delegate") -> AgentDecision:
    return AgentDecision(
        type=AgentDecisionType.DELEGATE,
        delegate_agent_id=agent_id,
        reason=reason,
    )
