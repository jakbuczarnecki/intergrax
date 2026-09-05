# © Artur Czarnecki. All rights reserved.

"""Deterministic Tier-2 agent for canonical lifecycle E2E proofs (Stage 15)."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.agents.reference_harness import (
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from testing_support.builder import MeteringFakeLLMAdapter
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext

CANONICAL_PING_CONTRACT_ID = "canonical-ping-agent"
CANONICAL_PING_CAPABILITY = "canonical.ping"
CANONICAL_PING_INPUT = "ping"
CANONICAL_PING_OUTPUT = "canonical-agent-ok"


class CanonicalPingAgent(ReflexAgent):
    """Offline deterministic agent for Stage 15 lifecycle proofs."""

    contract_id = CANONICAL_PING_CONTRACT_ID
    capabilities = (CANONICAL_PING_CAPABILITY,)
    agent_name = "Canonical Ping Agent"
    agent_description = "Returns a fixed response for canonical lifecycle proofs."
    agent_version = "1.0.0"
    risk_level = AgentRiskLevel.LOW
    max_steps = 1
    cognitive_pattern = CognitivePattern.REFLEX

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version=self.agent_version,
            capabilities=list(self.capabilities),
            skills=[],
            extra_tools=[],
            risk_level=self.risk_level,
            lifecycle_state=AgentLifecycleState.PRODUCTION,
            production_eligible=True,
            owner_team="platform",
            owner_contact="harness@intergrax",
            on_call_contact="harness@intergrax",
            runbook_ref="docs/project/architecture/AGENT_DISTRIBUTION.md",
            modality_profile_id="lab.default",
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            validation_rules=["structured_output"],
            max_steps=self.max_steps,
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, CANONICAL_PING_CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=[CANONICAL_PING_CAPABILITY],
                score=1.0,
                rationale="canonical lifecycle proof agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=MeteringFakeLLMAdapter(),
            harness=default_reference_harness(),
        )

    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        message = self.read_run_input(step_ctx)
        return Observation(summary=message or "")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        del step_ctx
        return ReasoningResult(thought=observation.summary)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        del step_ctx
        if reasoning.thought == CANONICAL_PING_INPUT:
            return {
                "summary": CANONICAL_PING_OUTPUT,
                "answer": CANONICAL_PING_OUTPUT,
            }
        return {
            "summary": f"unexpected-input:{reasoning.thought}",
            "answer": f"unexpected-input:{reasoning.thought}",
        }

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        del step_ctx, output
        return AgentEvaluation(
            verdict=CognitiveEvaluation.COMPLETE,
            reason="canonical_ping_goal_met",
        )


__all__ = [
    "CANONICAL_PING_CAPABILITY",
    "CANONICAL_PING_CONTRACT_ID",
    "CANONICAL_PING_INPUT",
    "CANONICAL_PING_OUTPUT",
    "CanonicalPingAgent",
]
