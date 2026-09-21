# © Artur Czarnecki. All rights reserved.

"""Capability routing extension for structural Tier-2 runtime agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.tier2_agent import Tier2Agent


class AgentRoutingContractError(RuntimeError):
    """Deterministic failure when routing requires ``RoutableTier2Agent``."""


@runtime_checkable
class RoutableTier2Agent(Tier2Agent, Protocol):
    """Tier-2 agent that may participate in capability-based Nexus selection."""

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult: ...


def require_routable_tier2_agent(agent: Tier2Agent) -> RoutableTier2Agent:
    """Fail closed when ``agent`` lacks the routing contract."""
    if isinstance(agent, RoutableTier2Agent):
        return agent
    agent_id = agent.get_contract().id
    raise AgentRoutingContractError(
        f"Agent '{agent_id}' does not implement RoutableTier2Agent (can_handle)"
    )


__all__ = [
    "AgentRoutingContractError",
    "RoutableTier2Agent",
    "require_routable_tier2_agent",
]
