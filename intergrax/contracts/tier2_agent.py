# © Artur Czarnecki. All rights reserved.

"""Tier-2 agent surface for Application binding contracts (no agents/ import)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult


@runtime_checkable
class Tier2Agent(Protocol):
    """Structural Tier-2 agent contract used by Application manifest bindings."""

    async def run(self, request: AgentRunRequest) -> AgentRunResult: ...

    def get_contract(self) -> AgentContract: ...


__all__ = ["Tier2Agent"]
