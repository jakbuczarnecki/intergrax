# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from abc import ABC, abstractmethod

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest


class Agent(ABC):
    """
    Tier-2 Agent contract.

    Agent is responsible for:
    - building RuntimeContext (including RuntimeConfig)
    - configuring pipeline via config.pipeline

    Agent is NOT responsible for:
    - RuntimeState
    - execution
    - lifecycle management
    """

    @abstractmethod
    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        """
        Build fully configured RuntimeContext for this agent.
        Must include:
        - RuntimeConfig
        - config.pipeline
        - all required dependencies
        """
        ...

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Execute this agent for the given request.

        Delegates to :meth:`intergrax.agents.agent_engine.AgentEngine.run_agent`
        so all Tier-2 agents share one runtime path.
        """
        from intergrax.agents.agent_engine import AgentEngine

        return await AgentEngine.run_agent(self, request)