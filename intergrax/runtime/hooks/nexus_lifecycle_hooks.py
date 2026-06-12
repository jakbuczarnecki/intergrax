# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus task lifecycle hooks via shared :class:`MiddlewarePipeline` (§42.20, B.06)."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.security.task_security_context import resource_tenant_id_for_task
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

APP_ENV_STATE_RUNTIME_KEY = TaskMetadataKey.APP_ENV_STATE
ENV_SNAPSHOT_RUNTIME_KEY = TaskMetadataKey.ENVIRONMENT_SNAPSHOT


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
            "tenant_id": task.tenant_id,
            "user_id": task.user_id,
            "resource_tenant_id": resource_tenant_id_for_task(task),
            "prompt": task.message,
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
        _merge_task_env_state(task, ctx)
        _merge_task_snapshot(task, ctx)
        result = await self._pipeline.run_before(point, ctx)
        _persist_task_snapshot(task, ctx)
        _persist_task_env_state(task, ctx)
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
        _merge_task_env_state(task, ctx)
        _merge_task_snapshot(task, ctx)
        result = await self._pipeline.run_after(point, ctx)
        _persist_task_snapshot(task, ctx)
        _persist_task_env_state(task, ctx)
        _guard(result, point)


def _merge_task_env_state(task: Task, ctx: HookContext) -> None:
    persisted = task.metadata.get(APP_ENV_STATE_RUNTIME_KEY)
    if isinstance(persisted, dict):
        ctx.runtime_state[APP_ENV_STATE_RUNTIME_KEY] = persisted


def _merge_task_snapshot(task: Task, ctx: HookContext) -> None:
    persisted = task.metadata.get(ENV_SNAPSHOT_RUNTIME_KEY)
    if isinstance(persisted, dict):
        ctx.runtime_state[ENV_SNAPSHOT_RUNTIME_KEY] = persisted
        profile_snapshot_id = persisted.get("profile_snapshot_id")
        if isinstance(profile_snapshot_id, str) and profile_snapshot_id:
            ctx.runtime_state["profile_snapshot_id"] = profile_snapshot_id


def _persist_task_snapshot(task: Task, ctx: HookContext) -> None:
    updated = ctx.runtime_state.get(ENV_SNAPSHOT_RUNTIME_KEY)
    if isinstance(updated, dict):
        task.metadata[ENV_SNAPSHOT_RUNTIME_KEY] = updated
        task.sync_metadata()


def _persist_task_env_state(task: Task, ctx: HookContext) -> None:
    updated = ctx.runtime_state.get(APP_ENV_STATE_RUNTIME_KEY)
    if isinstance(updated, dict):
        task.metadata[APP_ENV_STATE_RUNTIME_KEY] = updated
        task.sync_metadata()


def _guard(result: HookResult, point: HookPoint) -> None:
    if result.action != HookAction.ALLOW:
        raise NexusLifecycleHookError(
            result.reason or f"hook blocked at {point.value}: {result.action.value}"
        )
