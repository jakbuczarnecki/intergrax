# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from intergrax.agents.agent_contract import Agent
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.runtime_request_bridge import (
    acp_session_enabled,
    agent_run_result_to_runtime_answer,
    runtime_request_to_agent_run,
)
from intergrax.agents.uaep import UAEPBlockedError, UAEPExecutor
from intergrax.agents.uaep_protocol import supports_uaep
from intergrax.contracts.agent_run import AgentRunResult
from intergrax.contracts.validation import ValidationResult
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
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

    Resolves agents and executes via typed ACP session or UAEP bridge.
    Legacy pipeline fallback removed — ACP-only execution (ACP-CLOSE-LEG-5).
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
        production_mode: bool = False,
    ) -> None:
        if isinstance(agents, AgentRegistry):
            self._registry = agents
            self._agents = agents.as_dict()
        else:
            self._registry = None
            self._agents = agents
        self._production_mode = production_mode
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

    @property
    def shadow_manager(self) -> ShadowWorkspaceManager:
        return self._uaep.shadow_manager

    @property
    def sandbox_manager(self) -> SandboxSessionManager:
        return self._uaep.sandbox_manager

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        agent = self._resolve_agent(request)
        return await self._execute_agent_impl(agent, request, self._uaep, registry=self._registry)

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        agent = self._resolve_agent(request)
        contract = self._resolve_agent_contract(agent)
        run_id = request.run_id
        try:
            answer, validation, governance, structured_data = (
                await self._execute_agent_impl(
                    agent,
                    request,
                    self._uaep,
                    registry=self._registry,
                )
            )
        except UAEPBlockedError as exc:
            return AgentExecutionResult(
                agent_id=contract.id,
                run_id=run_id,
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[str(exc)],
            )
        execution = runtime_answer_to_agent_result(
            answer,
            agent_id=contract.id,
            valid=validation.valid,
            validation_errors=validation.errors,
            governance=governance,
        )
        if structured_data:
            execution.structured_data.update(structured_data)
        return execution

    def _resolve_agent_contract(self, agent: Agent):
        author_contract = agent.get_contract()
        if self._registry is not None and self._registry.has(author_contract.id):
            return self._registry.get_contract(author_contract.id)
        return author_contract

    def _resolve_agent(self, request: RuntimeRequest) -> Agent:
        agent_id = request.agent_id
        if not agent_id:
            raise ValueError("request.agent_id must be provided.")
        if self._registry is not None:
            if not self._registry.is_routable(
                agent_id,
                production_mode=self._production_mode,
            ):
                raise ValueError(f"Agent '{agent_id}' is not routable in current mode.")
            return self._registry.get(agent_id)
        agent = self._agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' is not registered.")
        return agent

    @staticmethod
    def _resolve_static_executor(
        uaep_executor: Optional[UAEPExecutor],
        event_bus: Optional[RuntimeEventBus],
    ) -> UAEPExecutor:
        if uaep_executor is not None:
            return uaep_executor
        if event_bus is not None:
            return UAEPExecutor(event_bus=event_bus)
        return _DEFAULT_UAEP

    @staticmethod
    async def run_agent(
        agent: Agent,
        request: RuntimeRequest,
        *,
        uaep_executor: Optional[UAEPExecutor] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        registry: AgentRegistry | None = None,
    ) -> RuntimeAnswer:
        executor = AgentEngine._resolve_static_executor(uaep_executor, event_bus)
        answer, _validation, _governance, _structured = await AgentEngine._execute_agent_impl(
            agent,
            request,
            executor,
            registry=registry,
        )
        return answer

    @staticmethod
    async def run_agent_with_result(
        agent: Agent,
        request: RuntimeRequest,
        *,
        uaep_executor: Optional[UAEPExecutor] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        registry: AgentRegistry | None = None,
    ) -> AgentExecutionResult:
        executor = AgentEngine._resolve_static_executor(uaep_executor, event_bus)
        contract = agent.get_contract()
        if registry is not None and registry.has(contract.id):
            contract = registry.get_contract(contract.id)
        run_id = request.run_id
        try:
            answer, validation, governance, structured_data = (
                await AgentEngine._execute_agent_impl(
                    agent,
                    request,
                    executor,
                    registry=registry,
                )
            )
        except UAEPBlockedError as exc:
            return AgentExecutionResult(
                agent_id=contract.id,
                run_id=run_id,
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[str(exc)],
            )
        execution = runtime_answer_to_agent_result(
            answer,
            agent_id=contract.id,
            valid=validation.valid,
            validation_errors=validation.errors,
            governance=governance,
        )
        if structured_data:
            execution.structured_data.update(structured_data)
        return execution

    @staticmethod
    async def _execute_agent_impl(
        agent: Agent,
        request: RuntimeRequest,
        uaep_executor: UAEPExecutor,
        *,
        registry: AgentRegistry | None = None,
    ) -> tuple[
        RuntimeAnswer,
        ValidationResult,
        Optional[GovernanceResolution],
        dict[str, Any],
    ]:
        if isinstance(agent, IntergraxAgent) and acp_session_enabled(request):
            contract = agent.get_contract()
            agent_run = runtime_request_to_agent_run(request, contract=contract)
            result = await agent.run(agent_run)
            if not isinstance(result, AgentRunResult):
                raise TypeError("IntergraxAgent.run must return AgentRunResult for ACP session")
            answer = agent_run_result_to_runtime_answer(result)
            validation = ValidationResult(
                valid=result.status.value == "succeeded",
                errors=[error.message for error in result.errors],
            )
            return answer, validation, None, dict(result.structured_data)

        if supports_uaep(agent):
            from intergrax.llm.messages import (
                model_input_messages_from_metadata,
                requires_structured_model_input,
                STRUCTURED_MODEL_INPUT_REQUIRED_REASON,
            )

            model_messages = model_input_messages_from_metadata(request.metadata)
            if requires_structured_model_input(model_messages):
                raise UAEPBlockedError(STRUCTURED_MODEL_INPUT_REQUIRED_REASON)
            author_contract = agent.get_contract()
            resolved_contract = author_contract
            if registry is not None and registry.has(author_contract.id):
                resolved_contract = registry.get_contract(author_contract.id)
            answer, validation, governance = await uaep_executor.execute(
                agent,
                request,
                contract=resolved_contract,
            )
            return answer, validation, governance, {}

        raise ValueError(
            f"{type(agent).__name__} is not executable: set acp.session.v1 metadata for "
            "IntergraxAgent ACP runs, implement UAEPAgent (get_steps/run_step), or migrate "
            "off legacy pipeline — use ACP on_next_step or UAEP test shim (ACP-CLOSE-LEG-5)."
        )
