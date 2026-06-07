# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dispute_intake.capabilities import CAPABILITIES
from dispute_intake.contract import build_agent_contract
from dispute_intake.steps.pipeline import build_pipeline, run_domain_step
from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.agents.uaep_pipeline import pipeline_agent_steps, pipeline_step_complete


class DisputeIntakeAgent(Agent):
    """UAEP-first scaffolded agent — replace domain logic in ``steps/`` and ``prompts/``."""

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id="dispute_intake",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        from intergrax.agents.defaults import harness_production_mode

        config = RuntimeConfig(
            llm_adapter=build_pipeline().llm_adapter,
            enable_rag=False,
            production_mode=harness_production_mode(),
            tenant_id=request.tenant_id,
        )
        config.pipeline = build_pipeline().pipeline
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(config=config, session_manager=session_manager)

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return pipeline_agent_steps(
            step_id="dispute_intake_step",
            step_name="dispute_intake_step",
            trace_label="dispute.intake",
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
        return pipeline_step_complete(reason="dispute_intake step finished")
