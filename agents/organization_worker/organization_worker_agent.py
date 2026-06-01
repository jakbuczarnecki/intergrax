# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Organization Worker — virtual worker inside Slack / Teams (§38).

Demonstrates long-running intake, HITL approval, and structured completion
without embedding orchestration in communication adapters.
"""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    HumanRequest,
    HumanRequestUrgency,
)
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.task.task import TaskContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

ORG_VENDOR_REPORT_CAPABILITY = "org.vendor_report"


class OrganizationWorkerAgent(Agent):
    """Prepares vendor reports and requests human approval before delivery."""

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="organization_worker",
            name="Organization Worker",
            description=(
                "Virtual organizational worker for vendor reports and coordinated reviews (§38)."
            ),
            version="1.0.0",
            capabilities=[ORG_VENDOR_REPORT_CAPABILITY],
            allowed_tools=[],
            risk_level=AgentRiskLevel.MEDIUM,
            max_steps=3,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, ORG_VENDOR_REPORT_CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id="organization_worker",
                matched_capabilities=[ORG_VENDOR_REPORT_CAPABILITY],
                score=1.0,
                rationale="organization worker capability",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="vendor report draft"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [
            AgentStep(
                step_id="prepare_vendor_report",
                step_name="prepare_vendor_report",
                step_index=0,
                trace_label=ORG_VENDOR_REPORT_CAPABILITY,
            )
        ]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        _ = step
        subject = (ctx.request.message if ctx.request else "") or "unspecified vendor"
        draft = (
            f"Draft vendor report prepared for: {subject.strip()}. "
            "Pending manager approval before distribution."
        )
        return StepOutput(
            step_id=step.step_id,
            summary=draft,
            data={
                "report_status": "draft",
                "subject": subject.strip(),
            },
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step
        if ctx.request and ctx.request.metadata.get("human_approved"):
            subject = (ctx.request.message if ctx.request else "") or "vendor report"
            if output and isinstance(output.data, dict) and output.data.get("subject"):
                subject = str(output.data["subject"])
            final = f"Vendor report for {subject.strip()} delivered to finance channel."
            ctx.metadata["runtime_answer"] = RuntimeAnswer(run_id=ctx.run_id, answer=final)
            return AgentDecision(
                type=AgentDecisionType.COMPLETE,
                reason="vendor report approved and sent",
                summary=final,
                data={"report_status": "sent", "subject": subject.strip()},
            )

        subject = (ctx.request.message if ctx.request else "") or "this vendor report"
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason="manager approval required before sending vendor report",
            human_request=HumanRequest(
                request_id="org_vendor_report_approval",
                prompt=f"Approve sending vendor report for {subject.strip()}?",
                options=["approve", "reject"],
                urgency=HumanRequestUrgency.HIGH,
                timeout_seconds=3600,
                default_on_timeout=AgentDecisionType.ESCALATE,
            ),
        )
