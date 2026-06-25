# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime middleware pipeline (architecture §42.20, §42.42)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.hook_runtime_guard import invoke_guarded_hook

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus


class MiddlewarePipeline:
    """
    Composes middleware and hooks around core runtime operations.

    before_* hooks run in ascending priority; after_* in descending (§42.20).
    """

    def __init__(
        self,
        hook_registry: Optional[HookRegistry] = None,
        middleware: Optional[List[RuntimeMiddleware]] = None,
        *,
        hook_timeout_seconds: float | None = None,
        event_bus: RuntimeEventBus | None = None,
    ) -> None:
        self._hooks = hook_registry or HookRegistry()
        self._middleware = sorted(middleware or [], key=lambda m: m.priority)
        self._hook_timeout_seconds = hook_timeout_seconds
        self._event_bus = event_bus

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks

    @property
    def hook_timeout_seconds(self) -> float | None:
        return self._hook_timeout_seconds

    def configure_hook_runtime(
        self,
        *,
        hook_timeout_seconds: float | None,
        event_bus: RuntimeEventBus | None,
    ) -> None:
        """Apply §32.6.5 wall-time cap and audit bus (APP-CON-5)."""
        self._hook_timeout_seconds = hook_timeout_seconds
        self._event_bus = event_bus

    async def _run_middleware_before(
        self,
        mw: RuntimeMiddleware,
        point: HookPoint,
        ctx: HookContext,
    ) -> HookResult:
        return await invoke_guarded_hook(
            hook_name=mw.name,
            point=point,
            ctx=ctx,
            coro_factory=lambda: mw.before(point, ctx),
            timeout_seconds=self._hook_timeout_seconds,
            event_bus=self._event_bus,
        )

    async def _run_middleware_after(
        self,
        mw: RuntimeMiddleware,
        point: HookPoint,
        ctx: HookContext,
    ) -> HookResult:
        return await invoke_guarded_hook(
            hook_name=mw.name,
            point=point,
            ctx=ctx,
            coro_factory=lambda: mw.after(point, ctx),
            timeout_seconds=self._hook_timeout_seconds,
            event_bus=self._event_bus,
        )

    async def run_before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        for mw in self._middleware:
            result = await self._run_middleware_before(mw, point, ctx)
            if result.action.value != "allow":
                return result
        return await invoke_guarded_hook(
            hook_name="hook_registry",
            point=point,
            ctx=ctx,
            coro_factory=lambda: self._hooks.run(point, ctx),
            timeout_seconds=self._hook_timeout_seconds,
            event_bus=self._event_bus,
        )

    async def run_after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        hook_result = await invoke_guarded_hook(
            hook_name="hook_registry",
            point=point,
            ctx=ctx,
            coro_factory=lambda: self._hooks.run(point, ctx),
            timeout_seconds=self._hook_timeout_seconds,
            event_bus=self._event_bus,
        )
        if hook_result.action.value != "allow":
            return hook_result
        for mw in reversed(self._middleware):
            result = await self._run_middleware_after(mw, point, ctx)
            if result.action.value != "allow":
                return result
        return HookResult()
