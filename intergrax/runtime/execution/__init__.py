# © Artur Czarnecki. All rights reserved.

from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionDelegate
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest

__all__ = [
    "Execution",
    "ExecutionBoundary",
    "ExecutionDelegate",
    "ExecutionCapability",
    "ExecutionRequest",
    "UnifiedTaskRunnerExecutionDelegate",
]


def __getattr__(name: str) -> object:
    if name == "UnifiedTaskRunnerExecutionDelegate":
        from intergrax.runtime.execution.task_compat import UnifiedTaskRunnerExecutionDelegate

        return UnifiedTaskRunnerExecutionDelegate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
