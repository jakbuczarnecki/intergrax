# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Callable, Optional

from intergrax.runtime.task.task import Task, TaskState


TransitionHandler = Callable[[Task], None]


class TaskLifecycle:
    """
    Explicit task state machine (§23).

    Transitions are logged via optional handler (e.g. trace emission).
    """

    _ALLOWED: dict[TaskState, set[TaskState]] = {
        TaskState.CREATED: {TaskState.CLASSIFIED, TaskState.FAILED, TaskState.CANCELLED},
        TaskState.CLASSIFIED: {TaskState.PLANNED, TaskState.FAILED, TaskState.CANCELLED},
        TaskState.PLANNED: {
            TaskState.RUNNING,
            TaskState.WAITING_FOR_RESOURCES,
            TaskState.WAITING_FOR_HUMAN,
            TaskState.FAILED,
            TaskState.CANCELLED,
        },
        TaskState.WAITING_FOR_RESOURCES: {
            TaskState.RUNNING,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.EXPIRED,
        },
        TaskState.WAITING_FOR_HUMAN: {
            TaskState.RUNNING,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.EXPIRED,
        },
        TaskState.RUNNING: {
            TaskState.VALIDATING,
            TaskState.WAITING_FOR_HUMAN,
            TaskState.FAILED,
            TaskState.CANCELLED,
        },
        TaskState.VALIDATING: {
            TaskState.COMPLETED,
            TaskState.PARTIALLY_COMPLETED,
            TaskState.NEEDS_MORE_INFORMATION,
            TaskState.FAILED,
            TaskState.CANCELLED,
        },
        TaskState.COMPLETED: set(),
        TaskState.PARTIALLY_COMPLETED: set(),
        TaskState.NEEDS_MORE_INFORMATION: set(),
        TaskState.FAILED: set(),
        TaskState.CANCELLED: set(),
        TaskState.EXPIRED: set(),
    }

    def __init__(self, *, on_transition: Optional[TransitionHandler] = None) -> None:
        self._on_transition = on_transition

    def transition(self, task: Task, new_state: TaskState) -> Task:
        allowed = self._ALLOWED.get(task.state, set())
        if new_state not in allowed and task.state != new_state:
            raise ValueError(
                f"Invalid task transition: {task.state.value} -> {new_state.value}"
            )
        old = task.state
        task.state = new_state
        if self._on_transition is not None and old != new_state:
            self._on_transition(task)
        return task

    @staticmethod
    def created(task: Task) -> Task:
        task.state = TaskState.CREATED
        return task
