# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus task lifecycle hooks via shared :class:`MiddlewarePipeline` (§42.20, B.06)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.security.task_security_context import resource_tenant_id_for_task
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

APP_ENV_STATE_RUNTIME_KEY = TaskMetadataKey.APP_ENV_STATE
ENV_SNAPSHOT_RUNTIME_KEY = TaskMetadataKey.ENVIRONMENT_SNAPSHOT
CAPABILITY_ALIAS_REDIRECT_KEY = TaskMetadataKey.CAPABILITY_ALIAS_REDIRECT


class NexusLifecycleHookError(RuntimeError):
    """Raised when a lifecycle middleware hook blocks execution."""


def nexus_lifecycle_hook_context(
    task: Task,
    *,
    phase: ExecutionPhase,
    extra: Optional[dict[str, Any]] = None,
) -> HookContext:
    run_id, _ = require_active_execution_identity()
    return HookContext(
        task_id=task.task_id,
        run_id=run_id,
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
        _persist_intake_capability(task, ctx, point=point)
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


def _persist_intake_capability(task: Task, ctx: HookContext, *, point: HookPoint) -> None:
    if point != HookPoint.BEFORE_TASK_INTAKE:
        return
    updated_capability = ctx.runtime_state.get("capability")
    if isinstance(updated_capability, str):
        resolved = updated_capability.strip()
        if resolved and resolved != (task.context.capability or ""):
            task.context = task.context.model_copy(update={"capability": resolved})
    redirect = ctx.runtime_state.get(CAPABILITY_ALIAS_REDIRECT_KEY)
    if isinstance(redirect, dict):
        task.metadata[CAPABILITY_ALIAS_REDIRECT_KEY] = redirect
        task.sync_metadata()


def _guard(result: HookResult, point: HookPoint) -> None:
    if result.action != HookAction.ALLOW:
        raise NexusLifecycleHookError(
            result.reason or f"hook blocked at {point.value}: {result.action.value}"
        )


def _lifecycle_hook_failure_kind(reason: str) -> str:
    if reason.startswith("hook_error:"):
        return "platform.hook.hook_error"
    if reason.startswith("hook_timeout:"):
        return "platform.hook.hook_timeout"
    return "platform.hook.hook_blocked"


def _lifecycle_hook_name_from_reason(reason: str) -> str:
    if reason.startswith(("hook_error:", "hook_timeout:")):
        rest = reason.split(":", 1)[1]
        return rest.split(":", 1)[0] or "lifecycle_hook"
    return "lifecycle_hook"


def _sanitized_lifecycle_hook_reason(reason: str, *, max_len: int = 500) -> str:
    text = reason.strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


async def publish_nexus_lifecycle_hook_failure(
    publish: Callable[..., Awaitable[None]],
    *,
    task: Task,
    point: HookPoint,
    phase: ExecutionPhase,
    error: NexusLifecycleHookError,
    non_critical: bool = False,
) -> None:
    """Emit a visible runtime event for lifecycle/finalization hook failures."""
    from intergrax.runtime.events.spine_consolidation import build_platform_signal_event

    reason = str(error)
    kind = _lifecycle_hook_failure_kind(reason)
    severity = (
        EventSeverity.ERROR
        if kind.endswith("hook_error") or kind.endswith("hook_timeout")
        else EventSeverity.WARNING
    )
    event = build_platform_signal_event(
        kind=kind,
        task_id=task.task_id,
        run_id=require_active_execution_identity()[0],
        tenant_id=task.tenant_id,
        agent_id=task.agent_id,
        phase=phase,
        severity=severity,
        correlation_id=task.task_id,
        payload={
            "hook_name": _lifecycle_hook_name_from_reason(reason),
            "point": point.value,
            "reason": _sanitized_lifecycle_hook_reason(reason),
            "error_type": type(error).__name__,
            "non_critical": non_critical,
        },
    )
    await publish(event, task=task)
