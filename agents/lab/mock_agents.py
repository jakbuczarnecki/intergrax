# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime validation agents — stress-test registration, UAEP, and routing (Phase L.4).

These agents implement no business value; they return deterministic stubs for harness tests.
"""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.applications._shared.runtime_defaults import harness_production_mode
from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep_pipeline import (
    pipeline_agent_steps,
    pipeline_step_complete,
    run_pipeline_step,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.task.task import TaskContext


class _StubLLM(LLMAdapter):
    def __init__(self, *, provider: str, prefix: str) -> None:
        super().__init__()
        self.provider = provider
        self.model = f"{provider}-stub"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        call = self.usage.begin_call(run_id=run_id)
        try:
            for msg in reversed(messages):
                content = msg.content or ""
                if content:
                    return f"{self.provider}: {content[:120]}"
            return f"{self.provider}: (empty)"
        finally:
            self.usage.end_call(call, input_tokens=0, output_tokens=1, success=True)


class _MockPipeline(RuntimePipeline):
    def __init__(self, *, prefix: str) -> None:
        self._prefix = prefix

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(
            [*SETUP_STEPS, PersistAndBuildAnswerStep()],
            state,
        )
        message = (state.request.message or "").strip()
        answer = f"{self._prefix}: {message}"
        if state.runtime_answer is not None:
            state.runtime_answer.answer = answer
        if state.runtime_answer is None:
            raise RuntimeError(f"{self._prefix} pipeline did not produce runtime_answer.")
        return state.runtime_answer


class _MockAgentBase(Agent):
    """UAEP pipeline-backed mock agent."""

    def __init__(
        self,
        *,
        agent_id: str,
        name: str,
        capability: str,
        prefix: str,
        provider: str,
    ) -> None:
        self._agent_id = agent_id
        self._name = name
        self._capability = capability
        self._prefix = prefix
        self._provider = provider

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._name,
            description=f"Runtime validation mock ({self._agent_id}).",
            version="0.1.0",
            capabilities=[self._capability],
            allowed_tools=[],
            risk_level=AgentRiskLevel.LOW,
            max_steps=5,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, self._capability):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
                rationale="mock capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=_StubLLM(provider=self._provider, prefix=self._prefix),
            enable_rag=False,
            production_mode=harness_production_mode(),
            tenant_id=request.tenant_id,
        )
        config.pipeline = _MockPipeline(prefix=self._prefix)
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(config=config, session_manager=session_manager)

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id=f"{self._agent_id}_pipeline",
            step_name=f"{self._agent_id}_pipeline",
            trace_label=self._capability,
            allowed_tools=list(contract.allowed_tools),
        )

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        return await run_pipeline_step(step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return pipeline_step_complete(reason=f"{self._agent_id} mock finished")


class ResearchMockAgent(_MockAgentBase):
    def __init__(self) -> None:
        super().__init__(
            agent_id="research_mock",
            name="Research Mock Agent",
            capability="lab.research_mock",
            prefix="research-mock",
            provider="research_mock",
        )


class DocumentMockAgent(_MockAgentBase):
    def __init__(self) -> None:
        super().__init__(
            agent_id="document_mock",
            name="Document Mock Agent",
            capability="lab.document_mock",
            prefix="document-mock",
            provider="document_mock",
        )


class ValidatorMockAgent(_MockAgentBase):
    def __init__(self) -> None:
        super().__init__(
            agent_id="validator_mock",
            name="Validator Mock Agent",
            capability="lab.validator_mock",
            prefix="validator-mock",
            provider="validator_mock",
        )


class ComposerMockAgent(_MockAgentBase):
    def __init__(self) -> None:
        super().__init__(
            agent_id="composer_mock",
            name="Composer Mock Agent",
            capability="lab.composer_mock",
            prefix="composer-mock",
            provider="composer_mock",
        )
