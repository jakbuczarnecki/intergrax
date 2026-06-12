# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from signoff_probe.capabilities import CAPABILITIES
from signoff_probe.contract import build_agent_contract
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class SignoffProbeAgent(ReflexAgent):
    """Harness sign-off probe — typed Reflex pattern (ACP-MIG-3)."""

    contract_id = "signoff_probe"
    capabilities = tuple(CAPABILITIES)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "signoff_probe_step"

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or default_reference_harness()

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id="signoff_probe",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=PrefixStubLLMAdapter(prefix="signoff_probe"),
            harness=self._harness,
        )

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        message = self.read_run_input(step_ctx)
        return Observation(summary=message or "(empty)")

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
        answer = f"signoff_probe: {reasoning.thought}"
        return {"summary": answer, "answer": answer, "run_id": step_ctx.run_id}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="signoff_probe_goal_met")
