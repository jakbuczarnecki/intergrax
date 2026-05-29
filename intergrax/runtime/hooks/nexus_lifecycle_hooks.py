# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus task lifecycle hooks via shared :class:`MiddlewarePipeline` (§42.20, B.06)."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task


class NexusLifecycleHookError(RuntimeError):
    """Raised when a lifecycle middleware hook blocks execution."""


def nexus_lifecycle_hook_context(
    task: Task,
    *,
    phase: ExecutionPhase,
    extra: Optional[dict[str, Any]] = None,
) -> HookContext:
    return HookContext(
        task_id=task.task_id,
        run_id=task.task_id,
        agent_id=task.agent_id,
        phase=phase,
        runtime_state={
            "task_state": task.state.value,
            "classification": task.classification,
            "capability": task.context.capability,
            **(extra or {}),
        },
    )


class NexusLifecycleHookCoordinator:
    """Runs intake / classification / planning / finalization hooks on NexusLoop."""

    def __init__(self, pipeline: MiddlewarePipeline) -> None:
        self._pipeline = pipeline

    @property
    def pipeline(self) -> MiddlewarePipeline:
        return self._pipeline

    async def before(
        self,
        point: HookPoint,
        task: Task,
        *,
        phase: ExecutionPhase,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        ctx = nexus_lifecycle_hook_context(task, phase=phase, extra=extra)
        result = await self._pipeline.run_before(point, ctx)
        _guard(result, point)

    async def after(
        self,
        point: HookPoint,
        task: Task,
        *,
        phase: ExecutionPhase,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        ctx = nexus_lifecycle_hook_context(task, phase=phase, extra=extra)
        result = await self._pipeline.run_after(point, ctx)
        _guard(result, point)


def _guard(result: HookResult, point: HookPoint) -> None:
    if result.action != HookAction.ALLOW:
        raise NexusLifecycleHookError(
            result.reason or f"hook blocked at {point.value}: {result.action.value}"
        )
