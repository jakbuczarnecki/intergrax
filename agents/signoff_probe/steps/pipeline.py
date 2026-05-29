# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from intergrax.agents.uaep_pipeline import run_pipeline_step
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS


class _SignoffProbeLLMStub(LLMAdapter):
    provider = "signoff_probe"
    model = "signoff_probe-stub"

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
                return f"signoff_probe: {content[:200]}"
        return "signoff_probe: (empty)"


class _SignoffProbePipeline(RuntimePipeline):
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        await RuntimeStepRunner.execute_pipeline(
            [*SETUP_STEPS, PersistAndBuildAnswerStep()],
            state,
        )
        message = (state.request.message or "").strip()
        answer = f"signoff_probe: {message}"
        if state.runtime_answer is not None:
            state.runtime_answer.answer = answer
        if state.runtime_answer is None:
            raise RuntimeError("signoff_probe pipeline did not produce runtime_answer.")
        return state.runtime_answer


@dataclass(frozen=True)
class PipelineBundle:
    llm_adapter: LLMAdapter
    pipeline: _SignoffProbePipeline


def build_pipeline() -> PipelineBundle:
    return PipelineBundle(
        llm_adapter=_SignoffProbeLLMStub(),
        pipeline=_SignoffProbePipeline(),
    )


async def run_domain_step(step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
    """Replace with multi-step domain logic as the agent grows."""
    _ = step
    return await run_pipeline_step(step, ctx)
