# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Optional, Union

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep import UAEPExecutor, supports_uaep
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence

_DEFAULT_UAEP = UAEPExecutor()


class AgentEngine:
    """
    Tier-2 → Tier-1 bridge (§42.19, §42.44).

    Resolves agents and executes via UAEP when ``get_steps`` / ``run_step`` are
    implemented; otherwise falls back to the legacy ``RuntimeEngine`` pipeline path.
    """

    def __init__(
        self,
        agents: Union[Dict[str, Agent], AgentRegistry],
        *,
        event_bus: Optional[RuntimeEventBus] = None,
        middleware: Optional[MiddlewarePipeline] = None,
        uaep_executor: Optional[UAEPExecutor] = None,
        policy_engine: PolicyEngine | RuntimePolicyEngine | None = None,
        shadow_manager: Optional[ShadowWorkspaceManager] = None,
        sandbox_manager: Optional[SandboxSessionManager] = None,
        task_memory_store: Optional[TaskMemoryPersistence] = None,
    ) -> None:
        if isinstance(agents, AgentRegistry):
            self._registry = agents
            self._agents = agents.as_dict()
        else:
            self._registry = None
            self._agents = agents
        self._uaep = uaep_executor or UAEPExecutor(
            middleware=middleware,
            event_bus=event_bus,
            policy_engine=coerce_policy_engine(policy_engine),
            shadow_manager=shadow_manager,
            sandbox_manager=sandbox_manager,
            task_memory_store=task_memory_store,
        )

    @property
    def registry(self) -> AgentRegistry | None:
        return self._registry

    @property
    def uaep_executor(self) -> UAEPExecutor:
        return self._uaep

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        agent = self._resolve_agent(request)
        return await self.run_agent(agent, request, uaep_executor=self._uaep)

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        agent = self._resolve_agent(request)
        return await self.run_agent_with_result(agent, request, uaep_executor=self._uaep)

    def _resolve_agent(self, request: RuntimeRequest) -> Agent:
        agent_id = request.agent_id
        if not agent_id:
            raise ValueError("request.agent_id must be provided.")
        if self._registry is not None:
            return self._registry.get(agent_id)
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' is not registered.")
        return agent

    @staticmethod
    async def run_agent(
        agent: Agent,
        request: RuntimeRequest,
        *,
        uaep_executor: Optional[UAEPExecutor] = None,
    ) -> RuntimeAnswer:
        answer, _validation, _context, _governance = await AgentEngine._execute_agent_impl(
            agent,
            request,
            uaep_executor or _DEFAULT_UAEP,
        )
        return answer

    @staticmethod
    async def run_agent_with_result(
        agent: Agent,
        request: RuntimeRequest,
        *,
        uaep_executor: Optional[UAEPExecutor] = None,
    ) -> AgentExecutionResult:
        answer, validation, _context, governance = await AgentEngine._execute_agent_impl(
            agent,
            request,
            uaep_executor or _DEFAULT_UAEP,
        )
        contract = agent.get_contract()
        return runtime_answer_to_agent_result(
            answer,
            agent_id=contract.id,
            valid=validation.valid,
            validation_errors=validation.errors,
            governance=governance,
        )

    @staticmethod
    async def _execute_agent_impl(
        agent: Agent,
        request: RuntimeRequest,
        uaep_executor: UAEPExecutor,
    ) -> tuple[RuntimeAnswer, ValidationResult, RuntimeContext, Optional[GovernanceResolution]]:
        if supports_uaep(agent):
            return await uaep_executor.execute(agent, request)

        context = agent.build_context(request)
        runtime = RuntimeEngine(context)
        answer = await runtime.run(request)
        validation = agent.validate(answer, context=context)
        if not validation.valid and validation.errors:
            if answer.route is not None:
                answer.route.extra.setdefault("agent_validation_errors", validation.errors)
        return answer, validation, context, None
