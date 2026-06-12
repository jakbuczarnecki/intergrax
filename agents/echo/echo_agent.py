# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Sequence

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
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE


class _EchoLLMAdapter(LLMAdapter):
    provider = "echo"
    model = "echo-stub"

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
        _ = temperature, max_tokens, run_id
        for msg in reversed(messages):
            content = msg.content or ""
            if content:
                return build_adapter_response(content=f"echo: {content}")
        return build_adapter_response(content="echo: (empty)")


class EchoAgent(ReflexAgent):
    """Harness echo agent — typed Reflex pattern (ACP-MIG-3)."""

    contract_id = "echo"
    capabilities = ("echo.basic",)
    agent_name = "Echo Agent"
    agent_description = "Echoes user input for runtime harness validation."
    agent_version = "1.0.0"
    risk_level = AgentRiskLevel.LOW
    max_steps = 5
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "echo_pipeline"

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or default_reference_harness()

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            skills=[HARNESS_TOOL_SMOKE],
            extra_tools=[],
            risk_level=self.risk_level,
            lifecycle_state=AgentLifecycleState.PRODUCTION,
            production_eligible=True,
            owner_team="platform",
            owner_contact="harness@intergrax",
            on_call_contact="harness@intergrax",
            runbook_ref="docs/intergrax_runtime_architecture.md",
            modality_profile_id="lab.default",
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            validation_rules=["structured_output"],
            max_steps=self.max_steps,
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, "echo.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=["echo.basic"],
                score=1.0,
                rationale="default harness agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=_EchoLLMAdapter(),
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
        echoed = f"echo: {reasoning.thought}"
        return {"summary": echoed, "answer": echoed, "run_id": step_ctx.run_id}

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx, output
        return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason="echo_goal_met")
