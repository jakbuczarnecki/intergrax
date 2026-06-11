# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Organization Worker — virtual worker inside Slack / Teams (§38).

Demonstrates long-running intake, HITL approval, and structured completion
without embedding orchestration in communication adapters.
"""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.agents.authoring.acp_stub_reflex import perceive_run_input, reason_passthrough
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import AgentEvaluation, CognitiveEvaluation
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    HumanRequest,
    HumanRequestUrgency,
)
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.task.task import TaskContext
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.contracts.agent_step_context import AgentStepContext


class _StubLLMAdapter(LLMAdapter):
    provider = "org_worker"
    model = "stub"

    def __init__(self, fixed_text: str) -> None:
        self._text = fixed_text

    @property
    def context_window_tokens(self) -> int:
        return 32_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content=self._text)


ORG_VENDOR_REPORT_CAPABILITY = "org.vendor_report"


class OrganizationWorkerAgent(ReflexAgent):
    """Prepares vendor reports and requests human approval before delivery (ACP-MIG-5)."""

    contract_id = "organization_worker"
    capabilities = (ORG_VENDOR_REPORT_CAPABILITY,)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "prepare_vendor_report"

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="organization_worker",
            name="Organization Worker",
            description=(
                "Virtual organizational worker for vendor reports and coordinated reviews (§38)."
            ),
            version="1.0.0",
            capabilities=[ORG_VENDOR_REPORT_CAPABILITY],
            skills=[],
            extra_tools=[],
            risk_level=AgentRiskLevel.MEDIUM,
            lifecycle_state=AgentLifecycleState.STAGING,
            owner_team="platform",
            max_steps=3,
            cognitive_pattern=self.cognitive_pattern,
            pattern_version=self.pattern_version,
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
            llm_adapter=_StubLLMAdapter(fixed_text="vendor report draft"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        subject = (reasoning.thought or "").strip() or "unspecified vendor"
        draft = (
            f"Draft vendor report prepared for: {subject}. "
            "Pending manager approval before distribution."
        )
        return {
            "summary": draft,
            "answer": draft,
            "report_status": "draft",
            "subject": subject,
            "run_id": step_ctx.run_id,
        }

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]) -> AgentEvaluation:
        _ = output
        exec_ctx = step_ctx.metadata.get("uaep_exec_ctx")
        if isinstance(exec_ctx, RuntimeExecutionContext):
            request = exec_ctx.request
            if request and request.metadata.get("human_approved"):
                return AgentEvaluation(
                    verdict=CognitiveEvaluation.COMPLETE,
                    reason="vendor report approved and sent",
                )
        return AgentEvaluation(
            verdict=CognitiveEvaluation.HUMAN,
            reason="manager approval required before sending vendor report",
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
            ctx.metadata["runtime_answer"] = RuntimeAnswer(
                run_id=ctx.run_id,
                answer=final,
            )
            return AgentDecision(
                type=AgentDecisionType.COMPLETE,
                reason="vendor report approved and sent",
                payload={
                    "summary": final,
                    "report_status": "sent",
                    "subject": subject.strip(),
                },
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
