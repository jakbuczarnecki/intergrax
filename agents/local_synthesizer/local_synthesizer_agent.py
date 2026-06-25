# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from local_synthesizer.capabilities import CAPABILITIES
from local_synthesizer.contract import build_agent_contract
from local_synthesizer.steps.synthesize_job import run_synthesize_job
from intergrax.agents.authoring.acp_stub_reflex import (
    build_agent_runtime_context,
    evaluate_complete,
    perceive_run_input,
    reason_passthrough,
)
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class LocalSynthesizerAgent(ReflexAgent):
    """LKW synthesizer agent — typed Reflex pattern (ACP-MIG-4)."""

    contract_id = "local_synthesizer"
    capabilities = tuple(CAPABILITIES)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "local_synthesizer_step"

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id="local_synthesizer",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_agent_runtime_context(request, PrefixStubLLMAdapter(prefix="local_synthesizer"))

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        _ = reasoning
        return await run_synthesize_job(step_ctx)

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]):
        return evaluate_complete(step_ctx, output, reason="local_synthesizer_goal_met")
