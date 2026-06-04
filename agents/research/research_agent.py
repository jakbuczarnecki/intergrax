# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.uaep_pipeline import pipeline_agent_steps, pipeline_step_complete
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from intergrax.agents.tool_enablement import ToolEnablementProfile, ToolWiringContextLike
from research.steps.pipeline import build_pipeline, run_domain_step


class ResearchAgent(HarnessReferenceAgent):
    """Prototype research agent — stub pipeline with optional catalog websearch."""

    def __init__(
        self,
        harness: LabHarnessContext | None = None,
        *,
        tool_profile: ToolEnablementProfile | None = None,
        tool_wiring_context: ToolWiringContextLike | None = None,
        enable_websearch: bool = False,
    ) -> None:
        self._harness = harness or default_reference_harness()
        self._tool_profile = tool_profile
        self._tool_wiring_context = tool_wiring_context
        self._enable_websearch = enable_websearch

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="research",
            name="Research Agent",
            description="Prototype agent producing stub research findings.",
            version="0.1.0",
            capabilities=["research.web_search", "research.pipeline"],
            skills=[RESEARCH_LITERATURE_SCAN],
            extra_tools=[],
            risk_level=AgentRiskLevel.LOW,
            max_steps=10,
            validation_rules=["non_empty_summary"],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = {"research.web_search", "research.pipeline"}
        if capability in supported or capability is None:
            return CapabilityMatchResult(
                matched=True,
                agent_id="research",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="research capability",
            )
        return CapabilityMatchResult(matched=False, rationale="not a research capability")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        built = build_pipeline()
        has_web = bool(
            self._enable_websearch
            and self._tool_profile
            and self._tool_profile.is_tool_enabled("websearch.query")
        )
        runtime_context = build_lab_agent_runtime_context(
            request=request,
            llm_adapter=built.llm_adapter,
            harness=self._harness,
            pipeline=built.pipeline,
            enable_websearch=has_web,
        )
        runtime_context.config.tool_profile = self._tool_profile
        runtime_context.config.tool_wiring_context = self._tool_wiring_context
        return runtime_context

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="research_pipeline",
            step_name="research_pipeline",
            trace_label="research.web_search",
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
        return pipeline_step_complete(reason="research pipeline finished")
