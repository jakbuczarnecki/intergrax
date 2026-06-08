# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.agent.contracts import (
    AgentGetContractInput,
    AgentGetContractOutput,
    AgentListAgentsInput,
    AgentListAgentsOutput,
)
from intergrax.tools.providers.agent.service import agent_get_contract, agent_list_agents


class AgentListAgentsHandler(ServiceToolHandler[AgentListAgentsInput, AgentListAgentsOutput]):
    _service = agent_list_agents


class AgentGetContractHandler(ServiceToolHandler[AgentGetContractInput, AgentGetContractOutput]):
    _service = agent_get_contract
