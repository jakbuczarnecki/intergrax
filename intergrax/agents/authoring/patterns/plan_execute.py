# © Artur Czarnecki. All rights reserved.

"""Plan-execute pattern — phased plan | execute | synthesize (ACP-4)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.states import PlanExecuteSessionState
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext


class PlanExecuteAgent(CognitiveAgent):
    """Phase machine: plan → execute → synthesize (architecture §26.4)."""

    cognitive_pattern = CognitivePattern.PLAN_EXECUTE
    main_step_id = "plan_execute_main"
    session_state_type: type[PlanExecuteSessionState] = PlanExecuteSessionState

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        if not isinstance(state, PlanExecuteSessionState):
            state = PlanExecuteSessionState.model_validate(state.model_dump())

        observation = await self.perceive(step_ctx)
        reasoning = await self.reason(step_ctx, observation)
        output = await self.act(step_ctx, reasoning)
        evaluation = self.evaluate(step_ctx, output)

        next_phase = self._advance_phase(state.phase)
        updated = state.model_copy(update={"phase": next_phase, "iteration": state.iteration + 1})
        delta = self.session_state_delta(updated, exclude={"schema_version", "state_version"})

        if evaluation.verdict == CognitiveEvaluation.COMPLETE or next_phase == "done":
            return StepOutcome.complete(
                output=output,
                state_delta=delta,
            )
        return StepOutcome.continue_with(delta)

    @staticmethod
    def _advance_phase(phase: str) -> str:
        if phase == "plan":
            return "execute"
        if phase == "execute":
            return "synthesize"
        return "done"
