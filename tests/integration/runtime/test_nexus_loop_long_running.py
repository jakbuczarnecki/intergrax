# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from testing_support.nexus_hitl_test_agent import (
    NexusBasicHitlTestAgent,
    prepare_nexus_hitl_resume_task,
)


class _HitlAgent(NexusBasicHitlTestAgent):
    def __init__(self) -> None:
        super().__init__(track_step_runs=True, extended_human_approval=True)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_long_running_task_saves_checkpoint_on_pause(tmp_path):
    NexusBasicHitlTestAgent.step_run_count = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=LoggingNotificationAdapter(),
    )
    runner = UnifiedTaskRunner(
        loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    run_id = mint_run_id()

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="multi-day monitor",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
        ),
    )

    paused = await runner.run_task(task, run_id=run_id)

    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert paused.summary.resume_token
    assert paused.summary.checkpoint_id
    assert paused.metadata.get("resume_token") == paused.summary.resume_token
    checkpoints = store.list_for_task(paused.task_id, "t1")
    assert len(checkpoints) == 1
    assert checkpoints[0].runtime is not None
    assert checkpoints[0].runtime.uaep_step_index == 0
    assert checkpoints[0].runtime.last_step_output is not None
    assert NexusBasicHitlTestAgent.step_run_count == 1
    assert any(
        e.event_type == RuntimeEventType.PAUSED for e in loop.event_bus.history
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_long_running_task_resumes_with_token(tmp_path):
    NexusBasicHitlTestAgent.step_run_count = 0
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        checkpoint_store=store,
        notification_adapter=LoggingNotificationAdapter(),
    )
    runner = UnifiedTaskRunner(
        loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    run_id = mint_run_id()

    paused = await runner.run_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="multi-day monitor",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        ),
        run_id=run_id,
    )
    token = paused.summary.resume_token
    assert token
    checkpoint = store.get_latest(paused.task_id, "t1")
    assert checkpoint is not None and checkpoint.runtime is not None
    resume_task = Task(
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
            human=TaskHumanInput(
                response_text="approve",
                verdict=HumanResponseVerdict.APPROVE,
            ),
        ),
        metadata={"human_approved": True, "resume_token": token},
    )
    prepare_nexus_hitl_resume_task(
        resume_task,
        loaded=checkpoint,
        run_id=run_id,
        human_approved=True,
    )
    completed = await runner.run_task(
        resume_task,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )

    assert completed.state == TaskState.COMPLETED
    assert NexusBasicHitlTestAgent.step_run_count == 1
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
