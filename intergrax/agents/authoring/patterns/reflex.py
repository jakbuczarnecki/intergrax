# © Artur Czarnecki. All rights reserved.

"""Reflex pattern — single perceive → act → complete (ACP-2)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.base import CognitiveAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext


class ReflexAgent(CognitiveAgent):
    """One-shot cognitive pattern (architecture §26.2)."""

    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "reflex_main"
    max_steps: int = 1

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        _ = step_ctx
        return Observation(summary="reflex_input")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        _ = step_ctx
        return ReasoningResult(thought=observation.summary)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        _ = step_ctx
        return {"summary": reasoning.thought, "status": "reflex_complete"}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="reflex_goal_met")
