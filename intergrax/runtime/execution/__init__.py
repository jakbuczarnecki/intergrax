# © Artur Czarnecki. All rights reserved.

"""Execution Engine package — lazy public exports to avoid import cycles with bridge submodules."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Execution",
    "ExecutionBoundary",
    "ExecutionDelegate",
    "ExecutionCapability",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionRuntime",
    "ExecutionStatus",
    "RootExecutionOptions",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Execution": ("intergrax.runtime.execution.facade", "Execution"),
    "ExecutionBoundary": ("intergrax.runtime.execution.boundary", "ExecutionBoundary"),
    "ExecutionDelegate": ("intergrax.runtime.execution.boundary", "ExecutionDelegate"),
    "ExecutionCapability": ("intergrax.runtime.execution.request", "ExecutionCapability"),
    "ExecutionRequest": ("intergrax.runtime.execution.request", "ExecutionRequest"),
    "ExecutionResult": ("intergrax.runtime.execution.result", "ExecutionResult"),
    "ExecutionRuntime": ("intergrax.runtime.execution.runtime", "ExecutionRuntime"),
    "ExecutionStatus": ("intergrax.runtime.execution.result", "ExecutionStatus"),
    "RootExecutionOptions": ("intergrax.runtime.execution.runtime", "RootExecutionOptions"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = target
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)


if TYPE_CHECKING:
    from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionDelegate
    from intergrax.runtime.execution.facade import Execution
    from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
    from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus
    from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionOptions
