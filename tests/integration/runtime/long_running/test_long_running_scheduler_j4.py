# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from intergrax.utils import attribute_access

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.long_running.scheduler import LongRunningScheduler, UnifiedTaskResumeExecutor
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.utils.time_provider import SystemTimeProvider
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _TimeoutFailAgent(Agent):
    runs = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_timeout_fail",
            name="HITL timeout fail",
            description="auto-fail on timeout",
            capabilities=["hitl.timeout_fail"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        cap = attribute_access.optional(task_context, "capability", None)
        if cap in (None, "hitl.timeout_fail"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_timeout_fail",
                matched_capabilities=["hitl.timeout_fail"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="no")

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
        _TimeoutFailAgent.runs += 1
        return StepOutput(step_id=step.step_id, summary="review")

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
                request_id="hr_timeout_fail",
                prompt="Approve?",
                options=["approve", "reject"],
                timeout_seconds=30,
                default_on_timeout=AgentDecisionType.FAIL,
            ),
        )


class _TimeoutEscalateAgent(Agent):
    runs = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_timeout_escalate",
            name="HITL timeout escalate",
            description="auto-escalate on timeout",
            capabilities=["hitl.timeout_escalate"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        cap = attribute_access.optional(task_context, "capability", None)
        if cap in (None, "hitl.timeout_escalate"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_timeout_escalate",
                matched_capabilities=["hitl.timeout_escalate"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="no")

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
        _TimeoutEscalateAgent.runs += 1
        return StepOutput(step_id=step.step_id, summary="review")

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
                request_id="hr_timeout_escalate",
                prompt="Approve?",
                options=["approve", "reject", "escalate"],
                timeout_seconds=30,
                default_on_timeout=AgentDecisionType.ESCALATE,
            ),
        )


class _DelayedResumeAgent(Agent):
    runs = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="delayed_resume",
            name="Delayed resume",
            description="completes after scheduled resume",
            capabilities=["hitl.delayed"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        cap = attribute_access.optional(task_context, "capability", None)
        if cap in (None, "hitl.delayed"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="delayed_resume",
                matched_capabilities=["hitl.delayed"],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="no")

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
        _DelayedResumeAgent.runs += 1
        return StepOutput(step_id=step.step_id, summary="review")

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
                request_id="hr_delayed",
                prompt="Approve?",
                options=["approve", "reject"],
            ),
        )


def _build_scheduler(tmp_path, agent: Agent) -> tuple[LongRunningScheduler, NexusLoop, SQLiteTaskCheckpointStore]:
    _TimeoutFailAgent.runs = 0
    _TimeoutEscalateAgent.runs = 0
    _DelayedResumeAgent.runs = 0
    registry = AgentRegistry()
    registry.register(agent)
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "scheduler_ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=LoggingNotificationAdapter(),
    )
    runner = UnifiedTaskRunner(loop)
    scheduler = LongRunningScheduler(
        store,
        UnifiedTaskResumeExecutor(runner),
        schedule_store=store,
        ledger=store,
        notification_adapter=LoggingNotificationAdapter(),
        poll_interval_seconds=0.01,
    )
    return scheduler, loop, store


@pytest.mark.asyncio
async def test_scheduler_enforces_human_timeout_fail(tmp_path) -> None:
    scheduler, loop, store = _build_scheduler(tmp_path, _TimeoutFailAgent())
    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="timeout fail case",
            context=TaskContext(capability="hitl.timeout_fail"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    expires_raw = paused.metadata["human_request_expires_at"]
    expires_at = datetime.fromisoformat(expires_raw)
    after_expiry = expires_at + timedelta(seconds=5)

    with patch.object(SystemTimeProvider, "utc_now", return_value=after_expiry):
        processed = await scheduler.tick(now=after_expiry)

    assert processed == 1
    assert _TimeoutFailAgent.runs == 1
    checkpoint = store.list_paused()[0]
    assert store.has_action(f"timeout:{checkpoint.checkpoint_id}")

    with patch.object(SystemTimeProvider, "utc_now", return_value=after_expiry):
        assert await scheduler.tick(now=after_expiry) == 0


@pytest.mark.asyncio
async def test_scheduler_enforces_human_timeout_escalate(tmp_path) -> None:
    scheduler, loop, store = _build_scheduler(tmp_path, _TimeoutEscalateAgent())
    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="timeout escalate case",
            context=TaskContext(capability="hitl.timeout_escalate"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    expires_at = datetime.fromisoformat(paused.metadata["human_request_expires_at"])
    after_expiry = expires_at + timedelta(seconds=5)

    with patch.object(SystemTimeProvider, "utc_now", return_value=after_expiry):
        processed = await scheduler.tick(now=after_expiry)

    assert processed == 1
    assert _TimeoutEscalateAgent.runs == 1
    checkpoint = store.get_latest(paused.task_id, "t1")
    assert checkpoint is not None
    restored = Task.model_validate(checkpoint.task_snapshot)
    assert restored.runtime.governance.escalation_level == 1
    assert restored.state == TaskState.WAITING_FOR_HUMAN


@pytest.mark.asyncio
async def test_scheduler_delayed_resume_with_auto_approve(tmp_path) -> None:
    scheduler, loop, store = _build_scheduler(tmp_path, _DelayedResumeAgent())
    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="delayed resume case",
            context=TaskContext(capability="hitl.delayed"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    token = paused.summary.resume_token
    assert token
    assert paused.state == TaskState.WAITING_FOR_HUMAN

    run_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    scheduler.schedule_resume(
        task_id=paused.task_id,
        tenant_id="t1",
        resume_token=token,
        run_at_utc=run_at.isoformat(),
        resume_metadata={"human_approved": True},
    )

    processed = await scheduler.tick(now=datetime.now(timezone.utc))
    assert processed == 1
    assert _DelayedResumeAgent.runs == 1
    due = store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert due == []
    assert await scheduler.tick(now=datetime.now(timezone.utc)) == 0
