# © Artur Czarnecki. All rights reserved.

"""Decomposition pattern — sub-question queue (ACP-5)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.states import DecompositionSessionState
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


class DecompositionAgent(CognitiveAgent):
    """Iterative sub-question decomposition (architecture §26.5)."""

    cognitive_pattern = CognitivePattern.DECOMPOSITION
    main_step_id = "decomposition_main"
    session_state_type = DecompositionSessionState

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        if not isinstance(state, DecompositionSessionState):
            state = DecompositionSessionState.model_validate(state.model_dump())

        if not state.pending_sub_questions and len(state.answered) >= state.max_sub_questions:
            return StepOutcome.fail(
                errors=[
                    AgentRunError(
                        code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                        message="decomposition sub-question budget exhausted",
                    )
                ],
                terminal_reason=TerminalReason.MAX_STEPS_EXCEEDED,
            )

        observation = await self.perceive(step_ctx)
        reasoning = await self.reason(step_ctx, observation)
        output = await self.act(step_ctx, reasoning)
        evaluation = self.evaluate(step_ctx, output)

        question_key = str(output.get("question") or f"q-{len(state.answered) + 1}")
        answered = dict(state.answered)
        answered[question_key] = str(output.get("answer") or output.get("summary") or "")
        pending = [q for q in state.pending_sub_questions if q != question_key]
        updated = state.model_copy(
            update={
                "answered": answered,
                "pending_sub_questions": pending,
                "iteration": state.iteration + 1,
            }
        )
        delta = self.session_state_delta(updated, exclude={"schema_version", "state_version"})
        return self._evaluation_to_outcome(evaluation, output, delta)
