# © Artur Czarnecki. All rights reserved.

"""ReAct pattern — reason / act loop with budget (ACP-3 · TOOL-ENG-6)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.react_budget import record_react_tool_calls, sync_react_budget
from intergrax.agents.authoring.patterns.states import ReActSessionState
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, CognitivePattern, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext


class ReActAgent(CognitiveAgent):
    """Bounded reason→act loop inside one harness session (architecture §26.3)."""

    cognitive_pattern = CognitivePattern.REACT
    main_step_id = "react_main"
    default_max_react_iterations: int = 8

    session_state_type = ReActSessionState

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        if not isinstance(state, ReActSessionState):
            state = ReActSessionState.model_validate(state.model_dump())
        if state.max_react_iterations <= 0:
            state = state.model_copy(update={"max_react_iterations": self.default_max_react_iterations})
        state = sync_react_budget(state, default_max_iterations=self.default_max_react_iterations)

        last_output: dict[str, object] = {}
        while state.react_iterations_used < state.max_react_iterations:
            observation = await self.perceive(step_ctx)
            reasoning = await self.reason(step_ctx, observation)
            last_output = await self.act(step_ctx, reasoning)
            evaluation = self.evaluate(step_ctx, last_output)
            tool_calls_delta = int(last_output.get("tool_calls", 0) or 0)
            state = state.model_copy(
                update={
                    "react_iterations_used": state.react_iterations_used + 1,
                    "last_thought": reasoning.thought,
                    "iteration": state.iteration + 1,
                }
            )
            state = record_react_tool_calls(state, tool_calls_delta)
            state = sync_react_budget(state, default_max_iterations=self.default_max_react_iterations)
            delta = self.session_state_delta(state, exclude={"schema_version", "state_version"})
            if evaluation.verdict != CognitiveEvaluation.CONTINUE:
                return self._evaluation_to_outcome(evaluation, last_output, delta)
            step_ctx = step_ctx.model_copy(update={"state_snapshot": self._embed_state(step_ctx, state)})

        return StepOutcome.fail(
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                    message="react iteration budget exhausted",
                )
            ],
            terminal_reason=TerminalReason.MAX_STEPS_EXCEEDED,
            state_delta=self.session_state_delta(state, exclude={"schema_version", "state_version"}),
        )

    @staticmethod
    def _embed_state(step_ctx: AgentStepContext, state: ReActSessionState) -> dict[str, object]:
        from intergrax.contracts.acp_state import ACP_STATE_KEY

        root = dict(step_ctx.state_snapshot)
        root[ACP_STATE_KEY] = state.model_dump(by_alias=True)
        return root
