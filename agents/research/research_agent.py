# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

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
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from intergrax.agents.tool_enablement import ToolEnablementProfile, ToolWiringContextLike
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter


class ResearchAgent(ReflexAgent):
    """Prototype research agent — typed Reflex pattern (ACP-MIG-3)."""

    contract_id = "research"
    capabilities = ("research.web_search", "research.pipeline")
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "research_pipeline"

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
            lifecycle_state=AgentLifecycleState.STAGING,
            owner_team="platform",
            max_steps=10,
            validation_rules=["non_empty_summary"],
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
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
        has_web = bool(
            self._enable_websearch
            and self._tool_profile
            and self._tool_profile.is_tool_enabled("websearch.query")
        )
        runtime_context = build_lab_agent_runtime_context(
            request=request,
            llm_adapter=PrefixStubLLMAdapter(prefix="research"),
            harness=self._harness,
            enable_websearch=has_web,
        )
        runtime_context.config.tool_profile = self._tool_profile
        runtime_context.config.tool_wiring_context = self._tool_wiring_context
        return runtime_context

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        query = self.read_run_input(step_ctx)
        return Observation(summary=query or "(empty)")

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
        query = reasoning.thought
        findings = (
            f"research findings for '{query[:120]}': "
            "[stub: source A — relevant snippet], "
            "[stub: source B — supporting detail]"
        )
        return {"summary": findings, "answer": findings, "run_id": step_ctx.run_id}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="research_goal_met")
