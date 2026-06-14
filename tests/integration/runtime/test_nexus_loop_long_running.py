# © Artur Czarnecki. All rights reserved.

from intergrax.utils import attribute_access
import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.runtime.events.runtime_event import RuntimeEventType
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _HitlAgent(Agent):
    step_run_count = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl",
            name="HITL Agent",
            description="requests human approval once",
            capabilities=["hitl.basic"],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, "hitl.basic"):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl",
                matched_capabilities=["hitl.basic"],
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
        _HitlAgent.step_run_count += 1
        return StepOutput(step_id=step.step_id, summary="pending review")

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
                request_id="hr_hitl_1",
                prompt="Approve this action?",
                options=["approve", "reject"],
            ),
        )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_long_running_task_saves_checkpoint_on_pause(tmp_path):
    _HitlAgent.step_run_count = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=LoggingNotificationAdapter(),
    )

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="multi-day monitor",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
        ),
    )

    paused = await loop.handle_task(task)

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert paused.summary.resume_token
    assert paused.summary.checkpoint_id
    assert paused.metadata.get("resume_token") == paused.summary.resume_token
    checkpoints = store.list_for_task(paused.task_id, "t1")
    assert len(checkpoints) == 1
    assert checkpoints[0].runtime is not None
    assert checkpoints[0].runtime.uaep_step_index == 0
    assert checkpoints[0].runtime.last_step_output is not None
    assert _HitlAgent.step_run_count == 1
    assert any(
        e.event_type == RuntimeEventType.PAUSED for e in loop.event_bus.history
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_long_running_task_resumes_with_token(tmp_path):
    _HitlAgent.step_run_count = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=LoggingNotificationAdapter(),
    )

    paused = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="multi-day monitor",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    token = paused.summary.resume_token
    assert token

    completed = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="multi-day monitor",
            context=TaskContext(capability="hitl.basic"),
            task_id=paused.task_id,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(
                    enabled=True,
                    resume_token=token,
                ),
            ),
            metadata={"human_approved": True, "resume_token": token},
        )
    )

    assert completed.state == TaskState.COMPLETED
    assert _HitlAgent.step_run_count == 1
    assert any(
        e.event_type == RuntimeEventType.RESUMED for e in loop.event_bus.history
    )


@pytest.mark.unit
@pytest.mark.gate
def test_classifier_marks_long_running():
    from intergrax.runtime.nexus.task_classifier import ClassifyingTaskClassifier

    registry = AgentRegistry()
    registry.register(_HitlAgent())
    classifier = ClassifyingTaskClassifier(registry)
    task = classifier.classify(
        Task(
            tenant_id="t1",
            user_id="u1",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        )
    )
    assert task.classification == TaskClassification.LONG_RUNNING.value
