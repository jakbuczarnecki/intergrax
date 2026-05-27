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
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class _SummaryLLMStub(LLMAdapter):
    provider = "research-summary"
    model = "summary-stub"

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
                return f"summary-draft: {content[:300]}"
        return "summary-draft: (empty)"


class SummaryPipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(
            [*SETUP_STEPS, PersistAndBuildAnswerStep()],
            state,
        )
        raw = (state.request.message or "").strip()
        if "--- prior agent outputs ---" in raw:
            _, _, prior = raw.partition("--- prior agent outputs ---")
            summary = f"summary: {prior.strip()[:800]}"
        else:
            summary = f"summary: {raw[:800]}"
        if state.runtime_answer is not None:
            state.runtime_answer.answer = summary
        if state.runtime_answer is None:
            raise RuntimeError("SummaryPipeline did not produce runtime_answer.")
        return state.runtime_answer


class SummaryAgent(Agent):
    """Summarizes prior agent outputs in a multi-agent research flow."""

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="research-summary",
            name="Research Summary Agent",
            description="Summarizes research findings from prior graph nodes.",
            version="0.1.0",
            capabilities=["research.summarize"],
            allowed_tools=[],
            risk_level=AgentRiskLevel.LOW,
            max_steps=5,
            validation_rules=["non_empty_summary"],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "research.summarize"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="research-summary",
                matched_capabilities=["research.summarize"],
                score=1.0,
                rationale="summary step",
            )
        return CapabilityMatchResult(matched=False, rationale="not summary capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=_SummaryLLMStub(),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = SummaryPipeline()
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(config=config, session_manager=session_manager)

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="summary_pipeline",
            step_name="summary_pipeline",
            trace_label="research.summarize",
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
        return pipeline_step_complete(reason="summary pipeline finished")
