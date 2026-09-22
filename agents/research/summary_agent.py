# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.runtime.nexus.agents.acp_stub_reflex import (
    evaluate_complete,
    perceive_run_input,
    reason_passthrough,
    summary_act_output,
)
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    default_reference_harness,
)
from intergrax.runtime.nexus.agents.reference_harness_runtime import (
    build_lab_agent_runtime_context,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from research.contract import build_agent_contract
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter


class SummaryAgent(ReflexAgent):
    """Summarizes prior agent outputs — typed Reflex pattern (ACP-MIG-4)."""

    contract_id = "research-summary"
    capabilities = ("research.summarize",)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "summary_pipeline"

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or default_reference_harness()

    def get_contract(self) -> AgentContract:
        return build_agent_contract(type(self))

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        capability = routing_capability_from_envelope(task)
        if capability in (None, "research.summarize"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="research-summary",
                matched_capabilities=["research.summarize"],
                score=1.0,
                rationale="summary step",
            )
        return CapabilityMatchResult(matched=False, rationale="not summary capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=PrefixStubLLMAdapter(prefix="summary"),
            harness=self._harness,
        )

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        return summary_act_output(step_ctx, reasoning)

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]):
        return evaluate_complete(step_ctx, output, reason="summary_goal_met")
