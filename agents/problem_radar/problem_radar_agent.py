# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_pipeline import pipeline_agent_steps, pipeline_step_complete
from intergrax.applications._shared.lab_harness_context import LabHarnessContext
from intergrax.applications._shared.lab_runtime_config import build_lab_agent_runtime_context
from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from problem_radar.capabilities import CAPABILITIES
from problem_radar.contract import build_agent_contract
from problem_radar.steps.pipeline import build_pipeline, run_domain_step


class ProblemRadarAgent(HarnessReferenceAgent):
    """Phase K.1 prototype — stub clustering with typed ``ProblemRadarOutput``."""

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or LabHarnessContext(
            policy_bundle=build_runtime_policy_bundle(),
        )

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id="problem_radar",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="problem radar capability",
            )
        return CapabilityMatchResult(matched=False, rationale="not a problem radar capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        built = build_pipeline()
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=built.llm_adapter,
            harness=self._harness,
            pipeline=built.pipeline,
            enable_websearch=True,
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="problem_radar_step",
            step_name="problem_radar_step",
            trace_label="problem_radar.scan",
            allowed_tools=list(contract.allowed_tools),
        )

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        return await run_domain_step(step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return pipeline_step_complete(reason="problem_radar scan finished")
