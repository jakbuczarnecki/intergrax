# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_pipeline import pipeline_agent_steps, pipeline_step_complete
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from vendor_discovery.capabilities import CAPABILITIES
from vendor_discovery.contract import build_agent_contract
from vendor_discovery.steps.pipeline import build_pipeline, run_domain_step


class VendorDiscoveryAgent(HarnessReferenceAgent):
    """Phase K.2 prototype — stub vendor shortlist with typed ``VendorDiscoveryOutput``."""

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
                agent_id="vendor_discovery",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="vendor discovery capability",
            )
        return CapabilityMatchResult(
            matched=False, rationale="not a vendor discovery capability"
        )

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
            step_id="vendor_discovery_step",
            step_name="vendor_discovery_step",
            trace_label="vendor_discovery.search",
            allowed_tools=list(contract.allowed_tools),
        )

    async def run_step(
        self, step: AgentStep, ctx: RuntimeExecutionContext
    ) -> StepOutput:
        return await run_domain_step(step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return pipeline_step_complete(reason="vendor discovery scan finished")
