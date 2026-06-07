# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Message bus integration contract — aliases queueing TaskQueue (§7.1.2, Phase M.2)."""

from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskSummary,
)

MessageBus = TaskQueue

__all__ = [
    "MessageBus",
    "TaskHandle",
    "TaskQueue",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    "TaskSummary",
]
