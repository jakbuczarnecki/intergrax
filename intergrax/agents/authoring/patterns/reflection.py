# © Artur Czarnecki. All rights reserved.

"""Reflection pattern — draft → critique → revise (ACP-6)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.states import ReflectionSessionState
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext


class ReflectionAgent(CognitiveAgent):
    """Draft / critique / revise loop (architecture §26.6)."""

    cognitive_pattern = CognitivePattern.REFLECTION
    main_step_id = "reflection_main"
    session_state_type = ReflectionSessionState

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        if not isinstance(state, ReflectionSessionState):
            state = ReflectionSessionState.model_validate(state.model_dump())

        observation = await self.perceive(step_ctx)
        reasoning = await self.reason(step_ctx, observation)
        output = await self.act(step_ctx, reasoning)
        evaluation = self.evaluate(step_ctx, output)

        next_phase = self._advance_phase(state.phase)
        updated = state.model_copy(
            update={
                "phase": next_phase,
                "draft": str(output.get("draft") or state.draft or ""),
                "critique": str(output.get("critique") or state.critique or ""),
                "iteration": state.iteration + 1,
            }
        )
        delta = self.session_state_delta(updated, exclude={"schema_version", "state_version"})

        if evaluation.verdict == CognitiveEvaluation.COMPLETE or next_phase == "done":
            return StepOutcome.complete(output=output, state_delta=delta)
        return StepOutcome.continue_with(delta)

    @staticmethod
    def _advance_phase(phase: str) -> str:
        if phase == "draft":
            return "critique"
        if phase == "critique":
            return "revise"
        return "done"
