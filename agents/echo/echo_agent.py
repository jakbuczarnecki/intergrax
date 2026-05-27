# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.capability import CapabilityMatchResult
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


class _EchoLLMAdapter:
    """Minimal stub LLM for harness runs without external providers."""

    def generate(self, messages, **kwargs) -> str:
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or ""
            if content:
                return f"echo: {content}"
        return "echo: (empty)"


class EchoPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(
            [*SETUP_STEPS, PersistAndBuildAnswerStep()],
            state,
        )
        message = (state.request.message or "").strip()
        if state.runtime_answer is not None:
            state.runtime_answer.answer = f"echo: {message}"
        if state.runtime_answer is None:
            raise RuntimeError("EchoPipeline did not produce runtime_answer.")
        return state.runtime_answer


class EchoAgent(Agent):
    """Minimal agent: echoes user input through Nexus pipeline."""

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="echo",
            name="Echo Agent",
            description="Echoes user input for runtime harness validation.",
            version="1.0.0",
            capabilities=["echo.basic"],
            allowed_tools=[],
            risk_level=AgentRiskLevel.LOW,
            max_steps=5,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "echo.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="echo",
                matched_capabilities=["echo.basic"],
                score=1.0,
                rationale="default harness agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=_EchoLLMAdapter(),  # type: ignore[arg-type]
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = EchoPipeline()
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(
            config=config,
            session_manager=session_manager,
        )
