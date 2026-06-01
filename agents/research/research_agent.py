# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Sequence

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
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.applications._shared.runtime_defaults import harness_production_mode
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class _ResearchLLMStub(LLMAdapter):
    provider = "research"
    model = "research-stub"

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
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or ""
            if content:
                return f"research-note: {content[:200]}"
        return "research-note: (empty)"


class ResearchPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(
            [*SETUP_STEPS, PersistAndBuildAnswerStep()],
            state,
        )
        query = (state.request.message or "").strip()
        findings = (
            f"research findings for '{query[:120]}': "
            "[stub: source A — relevant snippet], "
            "[stub: source B — supporting detail]"
        )
        if state.runtime_answer is not None:
            state.runtime_answer.answer = findings
        if state.runtime_answer is None:
            raise RuntimeError("ResearchPipeline did not produce runtime_answer.")
        return state.runtime_answer


class ResearchAgent(Agent):
    """Prototype research agent — stub pipeline with optional catalog websearch."""

    def __init__(
        self,
        *,
        tool_profile: ToolProfile | None = None,
        tool_wiring_context: ToolWiringContext | None = None,
        enable_websearch: bool = False,
    ) -> None:
        self._tool_profile = tool_profile
        self._tool_wiring_context = tool_wiring_context
        self._enable_websearch = enable_websearch

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="research",
            name="Research Agent",
            description="Prototype agent producing stub research findings.",
            version="0.1.0",
            capabilities=["research.web_search", "research.pipeline"],
            allowed_tools=["websearch.query", "rag.retrieve"],
            risk_level=AgentRiskLevel.LOW,
            max_steps=10,
            validation_rules=["non_empty_summary"],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = {"research.web_search", "research.pipeline"}
        if capability in supported or capability is None:
            return CapabilityMatchResult(
                matched=True,
                agent_id="research",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="research capability",
            )
        return CapabilityMatchResult(matched=False, rationale="not a research capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        has_web = bool(
            self._enable_websearch
            and self._tool_profile
            and self._tool_profile.is_tool_enabled("websearch.query")
        )
        config = RuntimeConfig(
            llm_adapter=_ResearchLLMStub(),
            enable_rag=False,
            enable_websearch=has_web,
            production_mode=harness_production_mode(),
            tenant_id=request.tenant_id,
            tool_profile=self._tool_profile,
            tool_wiring_context=self._tool_wiring_context,
        )
        config.pipeline = ResearchPipeline()
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(config=config, session_manager=session_manager)

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="research_pipeline",
            step_name="research_pipeline",
            trace_label="research.web_search",
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
        return pipeline_step_complete(reason="research pipeline finished")
