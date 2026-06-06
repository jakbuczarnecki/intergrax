# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from intergrax.agents.uaep_pipeline import run_pipeline_step
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import (
    PersistAndBuildAnswerStep,
)
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS


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
    ) -> LLMAdapterResponse:
        for msg in reversed(messages):
            content = msg.content or ""
            if content:
                return build_adapter_response(content=f"research-note: {content[:200]}")
        return build_adapter_response(content="research-note: (empty)")


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


@dataclass(frozen=True)
class PipelineBundle:
    llm_adapter: LLMAdapter
    pipeline: ResearchPipeline


def build_pipeline() -> PipelineBundle:
    return PipelineBundle(
        llm_adapter=_ResearchLLMStub(),
        pipeline=ResearchPipeline(),
    )


async def run_domain_step(step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
    return await run_pipeline_step(step, ctx)
