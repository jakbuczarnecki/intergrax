# © Artur Czarnecki. All rights reserved.

"""Reflection pattern — draft → critique → revise (ACP-6 · ACP-CLOSE-PAT-2)."""

from __future__ import annotations

from intergrax.agents.authoring.decision_gateway import verify_reflection_draft_with_decision
from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.states import ReflectionSessionState
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
from intergrax.contracts.decision_revision import DecisionRevisionDisposition
from intergrax.runtime.decision_flow import DecisionFlowHostAction


class ReflectionAgent(CognitiveAgent):
    """Draft / critique / revise loop with optional CVL gateway (architecture §26.6)."""

    cognitive_pattern = CognitivePattern.REFLECTION
    main_step_id = "reflection_main"
    session_state_type = ReflectionSessionState
    default_max_reflection_rounds: int = 3

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        state = self.load_session_state(step_ctx)
        if not isinstance(state, ReflectionSessionState):
            state = ReflectionSessionState.model_validate(state.model_dump())
        if state.max_reflection_rounds <= 0:
            state = state.model_copy(
                update={"max_reflection_rounds": self.default_max_reflection_rounds},
            )

        contract = self.get_contract()
        if state.phase == "critique" and state.draft:
            decision_outcome = await verify_reflection_draft_with_decision(
                step_ctx,
                contract=contract,
                draft=state.draft,
            )
            if decision_outcome is not None:
                rounds = state.reflection_rounds_used + 1
                updated = state.model_copy(
                    update={
                        "reflection_rounds_used": rounds,
                        "critique": decision_outcome.summary,
                    }
                )
                delta = self.session_state_delta(updated, exclude={"schema_version", "state_version"})
                output = {"draft": state.draft, "critique": decision_outcome.summary}
                if decision_outcome.passed:
                    return StepOutcome.complete(
                        output=output,
                        state_delta=delta,
                        terminal_reason=TerminalReason.GOAL_MET,
                    )
                if decision_outcome.host_action is DecisionFlowHostAction.PENDING_HUMAN:
                    return StepOutcome.pause_hitl(
                        decision_outcome.summary,
                        state_delta=delta,
                    )
                if rounds >= updated.max_reflection_rounds:
                    return StepOutcome.fail(
                        errors=[
                            AgentRunError(
                                code=AgentRunErrorCode.MAX_STEPS_EXCEEDED,
                                message="reflection decision rounds exhausted",
                            )
                        ],
                        terminal_reason=TerminalReason.MAX_STEPS_EXCEEDED,
                        state_delta=delta,
                    )
                if decision_outcome.host_action is DecisionFlowHostAction.BLOCK:
                    if (
                        decision_outcome.revision_decision is not None
                        and decision_outcome.revision_decision.disposition
                        is DecisionRevisionDisposition.ALLOWED
                    ):
                        revised = updated.model_copy(update={"phase": "revise"})
                        return StepOutcome.continue_with(
                            state_delta=self.session_state_delta(
                                revised,
                                exclude={"schema_version", "state_version"},
                            ),
                        )
                    return StepOutcome.fail(
                        errors=[
                            AgentRunError(
                                code=AgentRunErrorCode.VALIDATION_FAILED,
                                message=decision_outcome.summary,
                            )
                        ],
                        terminal_reason=TerminalReason.VALIDATION_FAILED,
                        state_delta=delta,
                    )

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
