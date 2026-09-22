# © Artur Czarnecki. All rights reserved.

"""Declarative agent contract for manifest roster resolution."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract

from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ORDER_ASSISTANT_AGENT_ID,
    ORDER_ASSISTANT_CAPABILITY,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import SCENARIO_TOOL_IDS


def build_agent_contract() -> AgentContract:
    return AgentContract(
        id=ORDER_ASSISTANT_AGENT_ID,
        name="AI Order Assistant",
        description="Production-capable order status and shipping assistant.",
        capabilities=[ORDER_ASSISTANT_CAPABILITY],
        allowed_tools=list(SCENARIO_TOOL_IDS),
    )


__all__ = ["build_agent_contract"]
