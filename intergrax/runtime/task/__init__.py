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

if TYPE_CHECKING:
    from intergrax.runtime.task.host_task_execution_run_adapter import HostTaskExecutionRunAdapter
    from intergrax.runtime.task.queued_host_task_execution_adapter import QueuedHostTaskExecutionAdapter
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
    from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
    from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

__all__ = [
    "HostTaskExecutionRunAdapter",
    "PersistingTaskTraceEmitter",
    "QueuedHostTaskExecutionAdapter",
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
    if name == "HostTaskExecutionRunAdapter":
        from intergrax.runtime.task.host_task_execution_run_adapter import HostTaskExecutionRunAdapter

        return HostTaskExecutionRunAdapter
    if name == "QueuedHostTaskExecutionAdapter":
        from intergrax.runtime.task.queued_host_task_execution_adapter import QueuedHostTaskExecutionAdapter

        return QueuedHostTaskExecutionAdapter
    if name == "create_nexus_celery_worker_app":
        from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app

        return create_nexus_celery_worker_app
    if name == "new_run_id":
        from intergrax.runtime.task.task_run_bridge import new_run_id

        return new_run_id
    if name == "task_from_execution_request":
        from intergrax.runtime.task.task_run_bridge import task_from_execution_request

        return task_from_execution_request
    if name == "task_from_runtime_request":
        from intergrax.runtime.task.task_run_bridge import task_from_runtime_request

        return task_from_runtime_request
    if name == "task_result_to_payload":
        from intergrax.runtime.task.task_run_bridge import task_result_to_payload

        return task_result_to_payload
    if name == "task_to_execution_payload":
        from intergrax.runtime.task.task_run_bridge import task_to_execution_payload

        return task_to_execution_payload
    if name == "PersistingTaskTraceEmitter":
        from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter

        return PersistingTaskTraceEmitter
    if name == "TaskTraceEmitter":
        from intergrax.runtime.task.task_trace import TaskTraceEmitter

        return TaskTraceEmitter
    if name == "lifecycle_with_persisting_trace":
        from intergrax.runtime.task.task_trace import lifecycle_with_persisting_trace

        return lifecycle_with_persisting_trace
    if name == "lifecycle_with_trace":
        from intergrax.runtime.task.task_trace import lifecycle_with_trace

        return lifecycle_with_trace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
