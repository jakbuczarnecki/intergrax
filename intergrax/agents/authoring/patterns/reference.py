# © Artur Czarnecki. All rights reserved.

"""Harness reference probes — one minimal agent per pattern (ACP-9..10)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.decomposition import DecompositionAgent
from intergrax.agents.authoring.patterns.plan_execute import PlanExecuteAgent
from intergrax.agents.authoring.patterns.react import ReActAgent
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.reflection import ReflectionAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_step_context import AgentStepContext


class PatternReflexProbe(ReflexAgent):
    contract_id = "pattern_reflex_probe"
    capabilities = ("harness.pattern.reflex",)
    agent_name = "Pattern Reflex Probe"
    risk_level = AgentRiskLevel.LOW


class PatternReActProbe(ReActAgent):
    contract_id = "pattern_react_probe"
    capabilities = ("harness.pattern.react",)
    agent_name = "Pattern ReAct Probe"
    risk_level = AgentRiskLevel.LOW
    default_max_react_iterations = 3

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        _ = step_ctx
        return Observation(summary="react_probe_input")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        _ = step_ctx
        return ReasoningResult(thought=f"think:{observation.summary}")

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        state = self.load_session_state(step_ctx)
        if state.iteration >= 1:
            return {"summary": reasoning.thought, "final": True}
        return {"summary": reasoning.thought, "final": False}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx
        if output.get("final"):
            return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="probe_done")
        return AgentEvaluation(verdict=CognitiveEvaluation.CONTINUE, reason="probe_continue")


class PatternPlanExecuteProbe(PlanExecuteAgent):
    contract_id = "pattern_plan_execute_probe"
    capabilities = ("harness.pattern.plan_execute",)
    agent_name = "Pattern Plan Execute Probe"
    risk_level = AgentRiskLevel.LOW

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        state = self.load_session_state(step_ctx)
        return Observation(summary=f"phase:{state.phase}")

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
        return {"summary": reasoning.thought}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        state = self.load_session_state(step_ctx)
        _ = step_ctx, output
        if state.phase == "synthesize":
            return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE)
        return AgentEvaluation(verdict=CognitiveEvaluation.CONTINUE)


class PatternDecompositionProbe(DecompositionAgent):
    contract_id = "pattern_decomposition_probe"
    capabilities = ("harness.pattern.decomposition",)
    agent_name = "Pattern Decomposition Probe"
    risk_level = AgentRiskLevel.LOW

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        _ = step_ctx
        return Observation(summary="decompose")

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
        return {"question": "root", "answer": reasoning.thought, "summary": reasoning.thought}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE)


class PatternReflectionProbe(ReflectionAgent):
    contract_id = "pattern_reflection_probe"
    capabilities = ("harness.pattern.reflection",)
    agent_name = "Pattern Reflection Probe"
    risk_level = AgentRiskLevel.LOW

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        _ = step_ctx
        return Observation(summary="reflect")

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
        state = self.load_session_state(step_ctx)
        return {"draft": reasoning.thought, "phase": state.phase}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        state = self.load_session_state(step_ctx)
        _ = step_ctx, output
        if state.phase == "revise":
            return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE)
        return AgentEvaluation(verdict=CognitiveEvaluation.CONTINUE)
