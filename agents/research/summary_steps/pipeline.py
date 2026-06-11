# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from intergrax.agents.authoring.uaep_pipeline_bridge import run_pipeline_step
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
    ) -> LLMAdapterResponse:
        for msg in reversed(messages):
            content = msg.content or ""
            if content:
                return build_adapter_response(content=f"summary-draft: {content[:300]}")
        return build_adapter_response(content="summary-draft: (empty)")


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


@dataclass(frozen=True)
class SummaryPipelineBundle:
    llm_adapter: LLMAdapter
    pipeline: SummaryPipeline


def build_summary_pipeline() -> SummaryPipelineBundle:
    return SummaryPipelineBundle(
        llm_adapter=_SummaryLLMStub(),
        pipeline=SummaryPipeline(),
    )


async def run_summary_domain_step(step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
    return await run_pipeline_step(step, ctx)
