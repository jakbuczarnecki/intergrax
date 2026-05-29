# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskIsolationOptions,
    TaskResultSummary,
    TaskRuntimeState,
)
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_run_bridge import (
    new_run_id,
    task_from_execution_request,
    task_from_runtime_request,
    task_result_to_payload,
    task_to_execution_payload,
)
from intergrax.runtime.task.task_trace import (
    PersistingTaskTraceEmitter,
    TaskTraceEmitter,
    lifecycle_with_persisting_trace,
    lifecycle_with_trace,
)

if TYPE_CHECKING:
    from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter
    from intergrax.runtime.task.queued_nexus_execution_adapter import QueuedNexusExecutionAdapter
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
    from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

__all__ = [
    "NexusTaskExecutionAdapter",
    "PersistingTaskTraceEmitter",
    "QueuedNexusExecutionAdapter",
    "Task",
    "TaskContext",
    "TaskExecutionOptions",
    "TaskIsolationOptions",
    "TaskLifecycle",
    "TaskResult",
    "TaskResultSummary",
    "TaskRuntimeState",
    "TaskState",
    "TaskTraceEmitter",
    "UnifiedTaskRunner",
    "create_nexus_celery_worker_app",
    "lifecycle_with_persisting_trace",
    "lifecycle_with_trace",
    "new_run_id",
    "task_from_execution_request",
    "task_from_runtime_request",
    "task_result_to_payload",
    "task_to_execution_payload",
]


def __getattr__(name: str):
    if name == "UnifiedTaskRunner":
        from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

        return UnifiedTaskRunner
    if name == "NexusTaskExecutionAdapter":
        from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter

        return NexusTaskExecutionAdapter
    if name == "QueuedNexusExecutionAdapter":
        from intergrax.runtime.task.queued_nexus_execution_adapter import QueuedNexusExecutionAdapter

        return QueuedNexusExecutionAdapter
    if name == "create_nexus_celery_worker_app":
        from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

        return create_nexus_celery_worker_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
