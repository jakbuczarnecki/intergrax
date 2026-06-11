# © Artur Czarnecki. All rights reserved.

"""CognitiveAgent ABC — perceive / reason / act / evaluate wired to on_next_step (ACP-1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    CognitivePattern,
    TerminalReason,
)
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.acp_state import AcpSessionState
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext

PATTERN_VERSION: str = "acp.v1"


class CognitiveAgent(IntergraxAgent, ABC):
    """
    Pattern base — authors implement domain hooks; framework wires ``on_next_step``.

    UAEP ``get_steps`` / ``run_step`` remain for bridge compatibility (single main step).
    """

    cognitive_pattern: ClassVar[CognitivePattern] = CognitivePattern.CUSTOM
    pattern_version: ClassVar[str] = PATTERN_VERSION
    main_step_id: ClassVar[str] = "cognitive_main"

    def get_contract(self) -> AgentContract:
        contract = super().get_contract()
        return contract.model_copy(
            update={
                "cognitive_pattern": self.cognitive_pattern,
                "pattern_version": self.pattern_version,
            }
        )

    @abstractmethod
    async def perceive(self, step_ctx: AgentStepContext) -> Observation: ...

    @abstractmethod
    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult: ...

    @abstractmethod
    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]: ...

    @abstractmethod
    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation: ...

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        observation = await self.perceive(step_ctx)
        reasoning = await self.reason(step_ctx, observation)
        output = await self.act(step_ctx, reasoning)
        evaluation = self.evaluate(step_ctx, output)
        delta = self.session_state_delta(state, exclude={"schema_version", "state_version"})
        return self._evaluation_to_outcome(evaluation, output, delta)

    def _evaluation_to_outcome(
        self,
        evaluation: AgentEvaluation,
        output: dict[str, object],
        state_delta: dict[str, object],
    ) -> StepOutcome:
        if evaluation.verdict == CognitiveEvaluation.COMPLETE:
            return StepOutcome.complete(
                output=output,
                terminal_reason=TerminalReason.GOAL_MET,
                state_delta=state_delta,
                confidence=evaluation.confidence,
            )
        if evaluation.verdict == CognitiveEvaluation.CONTINUE:
            return StepOutcome.continue_with(state_delta)
        if evaluation.verdict == CognitiveEvaluation.REPLAN:
            return StepOutcome.replan(state_delta, diagnostics={"reason": evaluation.reason})
        if evaluation.verdict == CognitiveEvaluation.HUMAN:
            return StepOutcome.pause_hitl(evaluation.reason or "human_required", state_delta=state_delta)
        return StepOutcome.fail(
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.INTERNAL_ERROR,
                    message=evaluation.reason or CognitiveEvaluation.FAIL.value,
                )
            ],
            terminal_reason=TerminalReason.ERROR,
            state_delta=state_delta,
        )

    def _ordered_step_ids(self) -> list[str]:
        return [self.main_step_id]

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return [
            AgentStep(
                step_id=self.main_step_id,
                step_name=self.main_step_id,
                step_index=0,
                trace_label=self.cognitive_pattern.value,
                allowed_tools=list(contract.allowed_tools),
            )
        ]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        from intergrax.agents.authoring.acp_uaep_shim import execute_cognitive_step_via_acp

        return await execute_cognitive_step_via_acp(self, step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output
        from intergrax.agents.authoring.acp_uaep_shim import decide_after_cognitive_step

        return decide_after_cognitive_step(ctx, default_reason=f"{self.cognitive_pattern.value} step finished")

    @staticmethod
    def read_run_input(step_ctx: AgentStepContext) -> str:
        raw = step_ctx.metadata.get(AcpRunContextKey.RUN_INPUT, "")
        if isinstance(raw, dict):
            return str(raw.get("message") or raw.get("summary") or raw)
        return str(raw or "").strip()
