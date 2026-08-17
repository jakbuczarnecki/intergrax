# © Artur Czarnecki. All rights reserved.

"""Canonical AGENT_DECISION governance enforcement bridge (G3B)."""

from __future__ import annotations

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.runtime.interrupts.handler import GovernanceResolution


def blocks_agent_decision_execution(resolution: GovernanceResolution) -> bool:
    """Return True when governed decision execution must not proceed."""
    return (
        resolution.should_block_execution
        or resolution.should_pause
        or resolution.should_fail
    )


def agent_decision_failure_from_resolution(
    resolution: GovernanceResolution,
) -> AgentDecision:
    """Map a blocked governance resolution to a terminal agent decision."""
    if resolution.should_pause:
        return resolution.agent_decision
    reason = resolution.policy_decision.reason or "policy_denied"
    if resolution.should_fail or resolution.agent_decision.type in {
        AgentDecisionType.FAIL,
        AgentDecisionType.CANCEL,
    }:
        return AgentDecision(
            type=AgentDecisionType.FAIL,
            reason=reason,
            payload={"policy_action": resolution.policy_decision.action.value},
        )
    return AgentDecision(
        type=AgentDecisionType.FAIL,
        reason=reason,
        payload={"policy_action": resolution.policy_decision.action.value},
    )
