# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE


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
            runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
            modality_profile_id="lab.default",
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            validation_rules=["structured_output"],
            max_steps=self.max_steps,
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
        )

    def can_handle(self, task_context: TaskEnvelope) -> CapabilityMatchResult:
        capability = routing_capability_from_envelope(task_context)
        if capability in (None, "echo.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=["echo.basic"],
                score=1.0,
                rationale="default harness agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

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
