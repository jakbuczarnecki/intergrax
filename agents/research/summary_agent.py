# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.authoring.acp_stub_reflex import (
    evaluate_complete,
    perceive_run_input,
    reason_passthrough,
    summary_act_output,
)
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from research.summary_steps.pipeline import build_summary_pipeline


class SummaryAgent(ReflexAgent):
    """Summarizes prior agent outputs — typed Reflex pattern (ACP-MIG-4)."""

    contract_id = "research-summary"
    capabilities = ("research.summarize",)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "summary_pipeline"

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or default_reference_harness()

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="research-summary",
            name="Research Summary Agent",
            description="Summarizes research findings from prior graph nodes.",
            version="0.1.0",
            capabilities=["research.summarize"],
            skills=[],
            extra_tools=[],
            risk_level=AgentRiskLevel.LOW,
            lifecycle_state=AgentLifecycleState.STAGING,
            owner_team="platform",
            max_steps=5,
            validation_rules=["non_empty_summary"],
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
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
        built = build_summary_pipeline()
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=built.llm_adapter,
            harness=self._harness,
            pipeline=built.pipeline,
        )

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        return summary_act_output(step_ctx, reasoning)

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]):
        return evaluate_complete(step_ctx, output, reason="summary_goal_met")
