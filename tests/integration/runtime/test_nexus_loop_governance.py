# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from testing_support.nexus_hitl_test_agent import (
    NexusBasicHitlTestAgent,
    prepare_nexus_hitl_resume_task,
)
from testing_support.nexus_lab_task_execution import run_lab_nexus_task

_HitlAgent = NexusBasicHitlTestAgent


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_pauses_for_human_request():
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    loop = NexusLoop(registry)

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
    )

    result = await run_lab_nexus_task(loop, task)

    assert result.state == TaskState.WAITING_FOR_HUMAN
    assert result.metadata.get("governance_human_request") is not None
    assert any(
        e.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        for e in loop.event_bus.history
    )


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_resumes_after_human_approval(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(registry, checkpoint_store=checkpoint_store)
    run_id = mint_run_id()
    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        ),
        run_id=run_id,
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    checkpoint = checkpoint_store.get_latest(paused.task_id, "t1")
    assert checkpoint is not None and checkpoint.runtime is not None
    resume_token = paused.summary.resume_token
    assert resume_token
    resume_task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
        task_id=paused.task_id,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True, resume_token=resume_token
            ),
            human=TaskHumanInput(
                response_text="approve",
                verdict=HumanResponseVerdict.APPROVE,
            ),
        ),
        metadata={"human_approved": True, "resume_token": resume_token},
    )
    prepare_nexus_hitl_resume_task(
        resume_task,
        loaded=checkpoint,
        run_id=run_id,
        human_approved=True,
    )
    result = await run_lab_nexus_task(
        loop,
        resume_task,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )

    assert result.state == TaskState.COMPLETED


@pytest.mark.unit
@pytest.mark.gate
def test_human_pause_coordinator_records_response():
    task = Task(tenant_id="t1", user_id="u1", message="x")
    HumanPauseCoordinator.record_human_response(task, "approve")
    assert HumanPauseCoordinator.is_resumed(task) is True
    assert HumanPauseCoordinator.verdict_from_task(task) == HumanResponseVerdict.APPROVE


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_fails_on_human_rejection(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    human_store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        human_decision_store=human_store,
        checkpoint_store=checkpoint_store,
    )

    run_id = mint_run_id()
    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        ),
        run_id=run_id,
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    checkpoint = checkpoint_store.get_latest(paused.task_id, "t1")
    assert checkpoint is not None and checkpoint.runtime is not None
    resume_token = paused.summary.resume_token
    assert resume_token
    reject_task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
        task_id=paused.task_id,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True, resume_token=resume_token
            ),
            human=TaskHumanInput(
                response_text="reject",
                verdict=HumanResponseVerdict.REJECT,
            ),
        ),
        metadata={"human_response": "reject", "resume_token": resume_token},
    )
    prepare_nexus_hitl_resume_task(
        reject_task,
        loaded=checkpoint,
        run_id=run_id,
        human_rejected=True,
    )
    rejected = await run_lab_nexus_task(
        loop,
        reject_task,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )
    assert rejected.state == TaskState.FAILED
    assert "human rejected" in (rejected.metadata.get("validation_errors") or [""])[0]
    decisions = human_store.list_for_task(paused.task_id, "t1")
    assert len(decisions) == 1
    assert decisions[0].verdict == HumanResponseVerdict.REJECT


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_escalates_and_persists(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    human_store = SQLiteHumanDecisionStore(db_path=tmp_path / "human.db")
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    loop = NexusLoop(
        registry,
        human_decision_store=human_store,
        checkpoint_store=checkpoint_store,
    )

    task_id = mint_task_id()
    run_id = mint_run_id()
    paused_escalate = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            task_id=task_id,
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        ),
        run_id=run_id,
    )
    assert paused_escalate.state == TaskState.WAITING_FOR_HUMAN
    checkpoint = checkpoint_store.get_latest(task_id, "t1")
    assert checkpoint is not None and checkpoint.runtime is not None
    resume_token = paused_escalate.summary.resume_token
    assert resume_token
    escalate_task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive action",
        context=TaskContext(capability="hitl.basic"),
        task_id=task_id,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True, resume_token=resume_token
            ),
            human=TaskHumanInput(
                response_text="escalate",
                verdict=HumanResponseVerdict.ESCALATE,
            ),
        ),
        metadata={"human_response": "escalate", "resume_token": resume_token},
    )
    prepare_nexus_hitl_resume_task(escalate_task, loaded=checkpoint, run_id=run_id)

    escalated = await run_lab_nexus_task(
        loop,
        escalate_task,
        run_id=run_id,
        attempt_id=checkpoint.runtime.attempt_id,
        resume_checkpoint=checkpoint,
    )
    assert escalated.state == TaskState.WAITING_FOR_HUMAN
    assert escalated.metadata.get("escalation_level") == 1
    assert any(
        e.event_type == RuntimeEventType.INTERRUPT_ESCALATED
        for e in loop.event_bus.history
    )
    records = human_store.list_for_task(task_id, "t1")
    assert len(records) == 1
    assert records[0].verdict == HumanResponseVerdict.ESCALATE
