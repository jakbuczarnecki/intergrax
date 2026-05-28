# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.agent_decision import HumanRequestUrgency
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _RecordingNotificationAdapter(LoggingNotificationAdapter):
    last_metadata: dict = {}

    async def notify(self, message) -> None:
        _RecordingNotificationAdapter.last_metadata = dict(message.metadata)
        await super().notify(message)


class _TimedHitlAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_timed",
            name="Timed HITL Agent",
            description="requests critical human approval with timeout",
            capabilities=["hitl.timed"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = getattr(task_context, "capability", None)
        if capability in (None, "hitl.timed"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_timed",
                matched_capabilities=["hitl.timed"],
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
        return StepOutput(step_id=step.step_id, summary="pending critical review")

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


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_propagates_human_request_v2_on_pause(tmp_path):
    _RecordingNotificationAdapter.last_metadata = {}
    registry = AgentRegistry()
    registry.register(_TimedHitlAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=_RecordingNotificationAdapter(),
    )

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="critical vendor change",
            context=TaskContext(capability="hitl.timed"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
            ),
        )
    )

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    gov = paused.metadata.get("governance_human_request") or {}
    assert gov.get("urgency") == "critical"
    assert gov.get("timeout_seconds") == 600
    assert gov.get("default_on_timeout") == "escalate"
    assert paused.metadata.get("human_request_expires_at")

    approval_event = next(
        e
        for e in loop.event_bus.history
        if e.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        and (e.payload.get("human_request") or {}).get("urgency") == "critical"
    )
    event_request = approval_event.payload.get("human_request") or {}
    assert event_request.get("urgency") == "critical"
    assert event_request.get("expires_at_utc")

    assert _RecordingNotificationAdapter.last_metadata.get("urgency") == "critical"
    assert _RecordingNotificationAdapter.last_metadata.get("timeout_seconds") == 600

    completed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="critical vendor change",
            context=TaskContext(capability="hitl.timed"),
            task_id=paused.task_id,
            metadata={"human_approved": True, "resume_token": paused.summary.resume_token},
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(
                    enabled=True,
                    resume_token=paused.summary.resume_token,
                ),
            ),
        )
    )
    assert completed.state == TaskState.COMPLETED
