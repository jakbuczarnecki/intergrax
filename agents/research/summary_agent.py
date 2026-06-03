# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_pipeline import pipeline_agent_steps, pipeline_step_complete
from intergrax.applications._shared.lab_harness_context import LabHarnessContext
from intergrax.applications._shared.lab_runtime_config import build_lab_agent_runtime_context
from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from research.summary_steps.pipeline import build_summary_pipeline, run_summary_domain_step


class SummaryAgent(HarnessReferenceAgent):
    """Summarizes prior agent outputs in a multi-agent research flow."""

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or LabHarnessContext(
            policy_bundle=build_runtime_policy_bundle(),
        )

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
            max_steps=5,
            validation_rules=["non_empty_summary"],
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

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="summary_pipeline",
            step_name="summary_pipeline",
            trace_label="research.summarize",
            allowed_tools=list(contract.allowed_tools),
        )

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        return await run_summary_domain_step(step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return pipeline_step_complete(reason="summary pipeline finished")
