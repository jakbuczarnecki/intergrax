# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness Tier A — HITL escalation notifies PagerDuty channel via runtime adapter."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskHumanInput, TaskLongRunningOptions
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _RecordingPagerDutyAdapter:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def notify(self, message: Any) -> None:
        self.messages.append(message)


class _HitlLongRunningAgent(Agent):
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

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [AgentStep(step_id="review", step_name="review", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
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
async def test_hitl_escalation_uses_pagerduty_notification_adapter(tmp_path: Any) -> None:
    adapter = _RecordingPagerDutyAdapter()
    registry = AgentRegistry()
    registry.register(_HitlLongRunningAgent())
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    loop = NexusLoop(registry, human_decision_store=store, notification_adapter=adapter)

    task_id = "task_harness_pd_1"
    await loop.handle_task(
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
        )
    )

    escalated = await loop.handle_task(
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
                human=TaskHumanInput(response_text="escalate"),
            ),
        )
    )

    assert escalated.state == TaskState.WAITING_FOR_HUMAN
    assert escalated.metadata.get("escalation_level") == 1
    assert len(adapter.messages) == 1
    assert adapter.messages[0].channel == "pagerduty"
    assert "Escalation" in adapter.messages[0].subject or "escalat" in adapter.messages[0].body.lower()
