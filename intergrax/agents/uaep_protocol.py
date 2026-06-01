# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""UAEP agent protocol (§42.5, Phase Q+-T.1)."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@runtime_checkable
class UAEPAgent(Protocol):
    """Agents executed by :class:`UAEPExecutor` must implement this protocol."""

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        ...

    def get_contract(self) -> AgentContract:
        ...

    def get_steps(self, context: RuntimeContext) -> List[AgentStep]:
        ...

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        ...


@runtime_checkable
class UAEPAgentWithResume(UAEPAgent, Protocol):
    async def resume_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
        cursor: dict,
    ) -> StepOutput:
        ...


@runtime_checkable
class UAEPAgentWithDecide(UAEPAgent, Protocol):
    def decide_after_step(
        self,
        step: AgentStep,
        output: Optional[StepOutput],
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        ...


def is_uaep_agent(agent: object) -> bool:
    return isinstance(agent, UAEPAgent)


def supports_uaep(agent: object) -> bool:
    """Public alias for registry and :class:`AgentEngine`."""
    return is_uaep_agent(agent)
