# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent factory protocol for Tier-3 application wiring (Phase N.2.1)."""

from __future__ import annotations

from typing import Callable, Protocol

from intergrax.agents.agent_contract import Agent

# Canonical signature: (ApplicationBuildContext, AgentBinding) -> Agent
# Types omitted here to avoid circular imports between contract modules.
AgentFactory = Callable[..., Agent]


class SupportsAgentFactory(Protocol):
    """Callable that builds a Tier-2 agent for a manifest binding."""

    def __call__(self, ctx: object, binding: object) -> Agent: ...
