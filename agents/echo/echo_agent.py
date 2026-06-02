# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.applications._shared.lab_harness_context import LabHarnessContext
from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications._shared.lab_runtime_config import build_lab_agent_runtime_context
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class _EchoLLMAdapter(LLMAdapter):
    """Minimal LLM adapter for harness runs without external providers."""

    provider = "echo"
    model = "echo-stub"

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
                    return f"echo: {content}"
            return "echo: (empty)"
        finally:
            self.usage.end_call(call, input_tokens=0, output_tokens=1, success=True)


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


class EchoAgent(HarnessReferenceAgent):
    """Minimal agent: echoes user input through Nexus pipeline."""

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or LabHarnessContext(
            policy_bundle=build_runtime_policy_bundle(),
        )

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="echo",
            name="Echo Agent",
            description="Echoes user input for runtime harness validation.",
            version="1.0.0",
            capabilities=["echo.basic"],
            allowed_tools=[],
            skill_ids=["harness.tool_smoke"],
            risk_level=AgentRiskLevel.LOW,
            max_steps=5,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
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
        context = build_lab_agent_runtime_context(
            request=request,
            llm_adapter=_EchoLLMAdapter(),
            harness=self._harness,
            pipeline=EchoPipeline(),
        )
        return context

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        """UAEP reference: single pipeline step (§42.32)."""
        _ = context
        contract = self.get_contract()
        return [
            AgentStep(
                step_id="echo_pipeline",
                step_name="echo_pipeline",
                step_index=0,
                trace_label="echo.basic",
                allowed_tools=list(contract.allowed_tools),
            )
        ]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        """Execute Nexus pipeline inside runtime-controlled step boundary."""
        request = ctx.request
        runtime_context = ctx.domain_context
        if request is None or runtime_context is None:
            raise RuntimeError("UAEP context missing request or domain_context.")

        runtime = RuntimeEngine(runtime_context)
        answer = await runtime.run(request)
        ctx.metadata["runtime_answer"] = answer

        message = (request.message or "").strip()
        echoed = f"echo: {message}"
        if answer.answer != echoed and answer.answer:
            echoed = answer.answer

        return StepOutput(
            step_id=step.step_id,
            summary=echoed,
            data={"run_id": answer.run_id or ctx.run_id},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="echo step finished",
        )
