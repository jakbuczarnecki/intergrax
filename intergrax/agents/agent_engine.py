# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Dict

from intergrax.agents.agent_contract import Agent
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest, RuntimeAnswer


class AgentEngine:
    """
    Thin orchestration layer (Tier-2 → Tier-1 bridge).

    Responsibilities:
    - resolve agent by agent_id
    - delegate RuntimeContext construction to agent
    - instantiate RuntimeEngine
    - execute runtime

    Does NOT:
    - modify context
    - understand pipeline
    - manage execution internals
    """

    def __init__(self, agents: Dict[str, Agent]) -> None:
        self._agents = agents

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        agent_id = request.agent_id

        if not agent_id:
            raise ValueError("request.agent_id must be provided.")

        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' is not registered.")

        return await AgentEngine.run_agent(agent, request)

    @staticmethod
    async def run_agent(agent: Agent, request: RuntimeRequest) -> RuntimeAnswer:
        context = agent.build_context(request)
        runtime = RuntimeEngine(context)
        return await runtime.run(request)