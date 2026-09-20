# © Artur Czarnecki. All rights reserved.

"""Canonical UAEP HITL test doubles for Nexus loop / OBS-DIAG qualification."""

from __future__ import annotations

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.contracts.agent_contract_meta import AgentContract
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
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from datetime import UTC, datetime

from intergrax.contracts.execution_identity import RunId
from intergrax.contracts.human_approver import (
    HumanApproverEvidence,
    local_development_approver_evidence,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import HumanApprovalResolution
from intergrax.utils import attribute_access
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


def _human_approved(ctx: RuntimeExecutionContext, *, extended: bool) -> bool:
    if ctx.request and ctx.request.metadata.get("human_approved"):
        return True
    if not extended:
        return False
    if ctx.request and ctx.request.metadata.get("human_decision") == "approve":
        return True
    return bool(ctx.metadata.get("human_approved"))


class NexusBasicHitlTestAgent(HarnessReferenceAgent):
    """Single human gate; completes when approval metadata is present."""

    step_run_count: int = 0

    def __init__(
        self,
        *,
        agent_id: str = "hitl",
        capability: str = "hitl.basic",
        human_request_id: str = "hr_hitl_1",
        track_step_runs: bool = False,
        extended_human_approval: bool = False,
        step_summary: str = "pending review",
    ) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._human_request_id = human_request_id
        self._track_step_runs = track_step_runs
        self._extended_human_approval = extended_human_approval
        self._step_summary = step_summary

    @property
    def runs(self) -> int:
        return NexusBasicHitlTestAgent.step_run_count

    @runs.setter
    def runs(self, value: int) -> None:
        NexusBasicHitlTestAgent.step_run_count = value

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name="HITL Agent",
            description="requests human approval once",
            capabilities=[self._capability],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, self._capability):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self) -> list[AgentStep]:
        return [AgentStep(step_id="review", step_name="review", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        if self._track_step_runs:
            NexusBasicHitlTestAgent.step_run_count += 1
        return StepOutput(step_id=step.step_id, summary=self._step_summary)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step
        if output is not None and _human_approved(ctx, extended=self._extended_human_approval):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        if _human_approved(ctx, extended=self._extended_human_approval):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason="approval required",
            human_request=HumanRequest(
                request_id=self._human_request_id,
                prompt="Approve this action?",
                options=["approve", "reject"],
            ),
        )


def prepare_nexus_hitl_resume_task(
    resume: Task,
    *,
    loaded: TaskCheckpoint,
    run_id: RunId,
    human_approved: bool = True,
    human_rejected: bool = False,
    approver: HumanApproverEvidence | None = None,
) -> None:
    """Bind approver + resolution evidence before durable HITL resume (test harness only)."""
    resolved_approver = approver or local_development_approver_evidence(
        tenant_id=resume.tenant_id,
        actor_id=resume.user_id or "operator",
    )
    resume.options.human.approver = resolved_approver
    if not human_approved and not human_rejected:
        return
    verdict = (
        HumanResponseVerdict.APPROVE if human_approved else HumanResponseVerdict.REJECT
    )
    pause_record = resume.runtime.governance.pause_record
    pause_id = pause_record.pause_id if pause_record is not None else "hr_hitl_pause"
    human_request_id = (
        pause_record.human_request_id if pause_record is not None else "hr_hitl_1"
    )
    execution_id = None
    if loaded.runtime is not None and loaded.runtime.execution_tree.entries:
        execution_id = loaded.runtime.execution_tree.entries[0].execution_id
    resume.runtime.governance.hitl_resolution = HumanApprovalResolution(
        task_id=resume.task_id,
        pause_id=pause_id,
        human_request_id=human_request_id,
        verdict=verdict,
        approver=resolved_approver,
        resolved_at=datetime.now(UTC).isoformat(),
        run_id=str(run_id),
        attempt_id=loaded.runtime.attempt_id if loaded.runtime is not None else None,
        execution_id=execution_id,
    )


class NexusTimedHitlTestAgent(NexusBasicHitlTestAgent):
    """HITL test double with HumanRequest v2 urgency / timeout fields."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="hitl_timed",
            capability="hitl.timed",
            human_request_id="hr_timed_1",
            step_summary="pending critical review",
        )

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_timed",
            name="Timed HITL Agent",
            description="requests critical human approval with timeout",
            capabilities=["hitl.timed"],
            max_steps=2,
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output
        if ctx.request and ctx.request.metadata.get("human_approved"):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason="critical approval required",
            human_request=HumanRequest(
                request_id="hr_timed_1",
                prompt="Approve critical vendor change?",
                options=["approve", "reject"],
                urgency=HumanRequestUrgency.CRITICAL,
                timeout_seconds=600,
                default_on_timeout=AgentDecisionType.ESCALATE,
            ),
        )
