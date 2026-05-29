# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.notifications.templates.partial_result import (
    PARTIAL_RESULT_TEMPLATE_ID,
    build_partial_result_notification_message,
    is_partial_result_templated_message,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_partial_result_notification_message() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="monitor vendors",
        context=TaskContext(capability="research.basic"),
        state=TaskState.WAITING_FOR_RESOURCES,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
        ),
    )
    task.runtime.orchestration.progress_message = "batch 2/5 processed"
    task.runtime.orchestration.resume_token = "rt_partial"
    task.runtime.orchestration.checkpoint_id = "ckpt_partial"
    task.sync_metadata()

    message = build_partial_result_notification_message(
        task,
        progress_message="batch 2/5 processed",
        channel="log",
        partial_payload={"batch_index": 2, "batch_total": 5},
        last_step_summary="processed vendor batch",
    )
    assert message.subject.startswith("Task progress")
    assert "batch 2/5 processed" in message.body
    assert message.metadata["template"] == PARTIAL_RESULT_TEMPLATE_ID
    assert message.metadata["partial_payload"]["batch_index"] == 2
    assert is_partial_result_templated_message(message)
