# © Artur Czarnecki. All rights reserved.

"""Base class for lab/product reference agents implementing UAEP (Phase U-Con.1)."""

from __future__ import annotations

from abc import abstractmethod

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep_protocol import UAEPAgent
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext


class HarnessReferenceAgent(Agent):
    """
    Tier-2 reference agent base — explicitly implements :class:`UAEPAgent`.

    Subclasses must provide UAEP step execution; registration may enforce
  ``isinstance(agent, UAEPAgent)`` when ``requires_uaep`` is set on the binding.
    """

    @abstractmethod
    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        ...

    @abstractmethod
    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        ...


def assert_uaep_reference_agent(agent: Agent) -> None:
    """Validate reference agent implements UAEP before registry insert."""
    if not isinstance(agent, UAEPAgent):
        raise TypeError(
            f"{type(agent).__name__} must implement UAEPAgent (get_steps, run_step)"
        )
