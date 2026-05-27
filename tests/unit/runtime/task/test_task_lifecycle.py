# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle


def _task() -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_task_lifecycle_allows_created_to_classified():
    lifecycle = TaskLifecycle()
    task = _task()

    lifecycle.transition(task, TaskState.CLASSIFIED)

    assert task.state == TaskState.CLASSIFIED


@pytest.mark.unit
@pytest.mark.gate
def test_task_lifecycle_rejects_invalid_transition():
    lifecycle = TaskLifecycle()
    task = _task()

    with pytest.raises(ValueError, match="Invalid task transition"):
        lifecycle.transition(task, TaskState.COMPLETED)


@pytest.mark.unit
@pytest.mark.gate
def test_task_lifecycle_invokes_on_transition_handler():
    seen: list[TaskState] = []

    def _handler(task: Task) -> None:
        seen.append(task.state)

    lifecycle = TaskLifecycle(on_transition=_handler)
    task = _task()

    lifecycle.transition(task, TaskState.CLASSIFIED)
    lifecycle.transition(task, TaskState.PLANNED)

    assert seen == [TaskState.CLASSIFIED, TaskState.PLANNED]
