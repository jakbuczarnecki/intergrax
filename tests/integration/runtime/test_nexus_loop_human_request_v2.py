# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.notifications.templates.hitl import HITL_PAUSE_TEMPLATE_ID
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskLongRunningOptions,
)
from testing_support.nexus_hitl_test_agent import NexusTimedHitlTestAgent
from testing_support.nexus_lab_task_execution import (
    resume_lab_nexus_hitl,
    run_lab_nexus_task,
)
from intergrax.contracts.execution_identity import mint_run_id


class _RecordingNotificationAdapter(LoggingNotificationAdapter):
    last_metadata: dict = {}
    last_body: str = ""

    async def notify(self, message) -> None:
        _RecordingNotificationAdapter.last_metadata = dict(message.metadata)
        _RecordingNotificationAdapter.last_body = message.body
        await super().notify(message)


_TimedHitlAgent = NexusTimedHitlTestAgent


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

    run_id = mint_run_id()
    paused = await run_lab_nexus_task(
        loop,
        Task(
            tenant_id="t1",
            user_id="u1",
            message="critical vendor change",
            context=TaskContext(capability="hitl.timed"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
            ),
        ),
        run_id=run_id,
    )

    assert paused.state == TaskState.WAITING_FOR_HUMAN

    approval_event = next(
        e
        for e in loop.event_bus.history
        if e.event_type == RuntimeEventType.HUMAN_APPROVAL_REQUESTED
        and (e.payload.get("human_request") or {}).get("urgency") == "critical"
    )
    event_request = approval_event.payload.get("human_request") or {}
    assert event_request.get("urgency") == "critical"
    assert event_request.get("timeout_seconds") == 600
    assert event_request.get("default_on_timeout") == "escalate"
    assert event_request.get("expires_at_utc")

    assert (
        _RecordingNotificationAdapter.last_metadata.get("template")
        == HITL_PAUSE_TEMPLATE_ID
    )
    assert event_request.get("timeout_seconds") == 600
    assert "reply with `approve`" in _RecordingNotificationAdapter.last_body
    assert "reply with `reject`" in _RecordingNotificationAdapter.last_body

    completed = await resume_lab_nexus_hitl(
        loop,
        paused=paused,
        checkpoint_store=store,
        tenant_id="t1",
        user_id="u1",
        message="critical vendor change",
        capability="hitl.timed",
        run_id=run_id,
    )
    assert completed.state == TaskState.COMPLETED
