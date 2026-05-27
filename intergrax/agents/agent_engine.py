# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Union

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry


class AgentEngine:
    """
    Thin orchestration layer (Tier-2 → Tier-1 bridge).

    Responsibilities:
    - resolve agent by agent_id (from dict or AgentRegistry)
    - delegate RuntimeContext construction to agent
    - instantiate RuntimeEngine
    - execute runtime

    Does NOT:
    - modify context
    - understand pipeline internals
    - manage execution lifecycle
    """

    def __init__(
        self,
        agents: Union[Dict[str, Agent], AgentRegistry],
    ) -> None:
        if isinstance(agents, AgentRegistry):
            self._registry = agents
            self._agents = agents.as_dict()
        else:
            self._registry = None
            self._agents = agents

    @property
    def registry(self) -> AgentRegistry | None:
        return self._registry

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        agent_id = request.agent_id

        if not agent_id:
            raise ValueError("request.agent_id must be provided.")

        if self._registry is not None:
            agent = self._registry.get(agent_id)
        else:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise ValueError(f"Agent '{agent_id}' is not registered.")

        return await AgentEngine.run_agent(agent, request)

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        """Execute agent and return canonical AgentExecutionResult (§14)."""
        agent_id = request.agent_id
        if not agent_id:
            raise ValueError("request.agent_id must be provided.")
        if self._registry is not None:
            agent = self._registry.get(agent_id)
        else:
            agent = self._agents.get(agent_id)
            if agent is None:
                raise ValueError(f"Agent '{agent_id}' is not registered.")
        return await AgentEngine.run_agent_with_result(agent, request)

    @staticmethod
    async def run_agent(agent: Agent, request: RuntimeRequest) -> RuntimeAnswer:
        answer, _validation, _context = await AgentEngine._execute_agent(agent, request)
        return answer

    @staticmethod
    async def run_agent_with_result(
        agent: Agent,
        request: RuntimeRequest,
    ) -> AgentExecutionResult:
        answer, validation, _context = await AgentEngine._execute_agent(agent, request)
        contract = agent.get_contract()
        return runtime_answer_to_agent_result(
            answer,
            agent_id=contract.id,
            valid=validation.valid,
            validation_errors=validation.errors,
        )

    @staticmethod
    async def _execute_agent(
        agent: Agent,
        request: RuntimeRequest,
    ) -> tuple[RuntimeAnswer, ValidationResult, RuntimeContext]:
        context = agent.build_context(request)
        runtime = RuntimeEngine(context)
        answer = await runtime.run(request)
        validation = agent.validate(answer, context=context)
        if not validation.valid and validation.errors:
            if answer.route is not None:
                answer.route.extra.setdefault("agent_validation_errors", validation.errors)
        return answer, validation, context
