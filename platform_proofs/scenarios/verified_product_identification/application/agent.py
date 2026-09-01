"""Minimal scenario agent skeleton — implement domain behavior."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.task.task import TaskContext


class VerifiedProductIdentificationAgent(ReflexAgent):
    """TODO: implement scenario agent contract."""

    contract_id = "verified_product_identification"
    capabilities = ("verified_product_identification.run",)

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability in (None, "verified_product_identification.run"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="verified_product_identification",
                matched_capabilities=["verified_product_identification.run"],
                score=1.0,
                rationale="scenario skeleton agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        raise NotImplementedError("Implement scenario perception.")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        raise NotImplementedError("Implement scenario reasoning.")

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        raise NotImplementedError("Implement scenario action.")

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        raise NotImplementedError("Implement scenario evaluation.")
