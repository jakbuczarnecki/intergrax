# © Artur Czarnecki. All rights reserved.

"""Harness reference probes — one minimal agent per pattern (ACP-9..10)."""

from __future__ import annotations

from typing import Optional, Sequence

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
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class _ProbeLLMStub(LLMAdapter):
    provider = "pattern_probe"
    model = "pattern-probe-stub"

    def __init__(self, *, fixed_text: str) -> None:
        self._fixed_text = fixed_text

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
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content=self._fixed_text)


def _probe_context(request: RuntimeRequest, *, label: str) -> RuntimeContext:
    config = RuntimeConfig(
        llm_adapter=_ProbeLLMStub(fixed_text=label),
        enable_rag=False,
        production_mode=False,
        tenant_id=request.tenant_id,
    )
    return RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
    )


class PatternReflexProbe(ReflexAgent):
    contract_id = "pattern_reflex_probe"
    capabilities = ("harness.pattern.reflex",)
    agent_name = "Pattern Reflex Probe"
    risk_level = AgentRiskLevel.LOW

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _probe_context(request, label="reflex-probe")


class PatternReActProbe(ReActAgent):
    contract_id = "pattern_react_probe"
    capabilities = ("harness.pattern.react",)
    agent_name = "Pattern ReAct Probe"
    risk_level = AgentRiskLevel.LOW
    default_max_react_iterations = 3

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _probe_context(request, label="react-probe")

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

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _probe_context(request, label="plan-probe")

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

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _probe_context(request, label="decomposition-probe")

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

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _probe_context(request, label="reflection-probe")

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
