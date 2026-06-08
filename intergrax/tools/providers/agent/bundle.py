# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.agent.contracts import (
    AgentGetContractInput,
    AgentGetContractOutput,
    AgentListAgentsInput,
    AgentListAgentsOutput,
)
from intergrax.tools.providers.agent.handlers import AgentGetContractHandler, AgentListAgentsHandler
from intergrax.tools.providers.agent.service import AGENT_GET_CONTRACT_TOOL_ID, AGENT_LIST_AGENTS_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

AGENT_BUNDLE_ID = "agent"
AGENT_TOOL_IDS: tuple[str, ...] = (AGENT_LIST_AGENTS_TOOL_ID, AGENT_GET_CONTRACT_TOOL_ID)


def register_agent_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=AGENT_LIST_AGENTS_TOOL_ID,
            name=AGENT_LIST_AGENTS_TOOL_ID,
            description="List agent_ids registered in the host AgentRegistry.",
            description_short="List agents.",
            input_schema=AgentListAgentsInput,
            output_schema=AgentListAgentsOutput,
            error_mapping={},
            side_effects=False,
            category="agent",
            risk_level=ToolRiskLevel.LOW,
            tags=("agent", "introspection", "dx"),
        ),
        AgentListAgentsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=AGENT_GET_CONTRACT_TOOL_ID,
            name=AGENT_GET_CONTRACT_TOOL_ID,
            description="Return serialized AgentContract metadata for one registered agent.",
            description_short="Get agent contract.",
            input_schema=AgentGetContractInput,
            output_schema=AgentGetContractOutput,
            error_mapping={},
            side_effects=False,
            category="agent",
            risk_level=ToolRiskLevel.LOW,
            tags=("agent", "introspection", "dx"),
        ),
        AgentGetContractHandler(ctx),
    )
