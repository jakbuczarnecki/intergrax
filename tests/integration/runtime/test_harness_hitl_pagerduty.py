# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness Tier A — HITL escalation notifies PagerDuty channel via runtime adapter."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from testing_support.nexus_lab_task_execution import (
    resume_lab_nexus_hitl,
    run_lab_nexus_task,
)

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.agent_decision import (
    AgentDecision,
    AgentDecisionType,
    HumanRequest,
)
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskLongRunningOptions,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _RecordingPagerDutyAdapter:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def notify(self, message: Any) -> None:
        self.messages.append(message)


class _HitlLongRunningAgent(HarnessReferenceAgent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_lr",
            name="HITL Long Running",
            description="HITL with long-running options",
            capabilities=["hitl.lr"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, "hitl.lr"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_lr",
                matched_capabilities=["hitl.lr"],
                score=1.0,
            )
        return CapabilityMatchResult(
            matched=False, rationale="capability not supported"
        )

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

    async def run_step(
        self, step: AgentStep, ctx: RuntimeExecutionContext
    ) -> StepOutput:
        return StepOutput(step_id=step.step_id, summary="pending")

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
            reason="approval required",
            human_request=HumanRequest(
                request_id="hr_lr_1",
                prompt="Approve?",
                options=["approve", "reject", "escalate"],
            ),
        )


@pytest.mark.asyncio
async def test_hitl_escalation_uses_pagerduty_notification_adapter(
    tmp_path: Any,
) -> None:
    adapter = _RecordingPagerDutyAdapter()
    registry = AgentRegistry()
    registry.register(_HitlLongRunningAgent())
    human_store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        human_decision_store=human_store,
        checkpoint_store=checkpoint_store,
        notification_adapter=adapter,
    )

    task_id = mint_task_id()
    run_id = mint_run_id()
    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive",
            context=TaskContext(capability="hitl.lr"),
            task_id=task_id,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(
                    enabled=True,
                    notify_channel="pagerduty",
                ),
            ),
        ),
        run_id=run_id,
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    escalated = await resume_lab_nexus_hitl(
        loop,
        paused=paused,
        checkpoint_store=checkpoint_store,
        tenant_id="t1",
        user_id="u1",
        message="sensitive",
        capability="hitl.lr",
        run_id=run_id,
        verdict=HumanResponseVerdict.ESCALATE,
    )

    assert escalated.state == TaskState.WAITING_FOR_HUMAN
    assert escalated.metadata.get("escalation_level") == 1
    assert len(adapter.messages) >= 1
    assert adapter.messages[0].channel == "pagerduty"
    assert (
        "Escalation" in adapter.messages[0].subject
        or "escalat" in adapter.messages[0].body.lower()
    )
