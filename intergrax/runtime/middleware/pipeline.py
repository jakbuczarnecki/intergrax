# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime middleware pipeline (architecture §42.20, §42.42)."""

from __future__ import annotations

from typing import List, Optional

from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.middleware.base import RuntimeMiddleware


class MiddlewarePipeline:
    """
    Composes middleware and hooks around core runtime operations.

    before_* hooks run in ascending priority; after_* in descending (§42.20).
    """

    def __init__(
        self,
        hook_registry: Optional[HookRegistry] = None,
        middleware: Optional[List[RuntimeMiddleware]] = None,
    ) -> None:
        self._hooks = hook_registry or HookRegistry()
        self._middleware = sorted(middleware or [], key=lambda m: m.priority)

    @property
    def hooks(self) -> HookRegistry:
        return self._hooks

    async def run_before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        for mw in self._middleware:
            result = await mw.before(point, ctx)
            if result.action.value != "allow":
                return result
        return await self._hooks.run(point, ctx)

    async def run_after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        hook_result = await self._hooks.run(point, ctx)
        if hook_result.action.value != "allow":
            return hook_result
        for mw in reversed(self._middleware):
            result = await mw.after(point, ctx)
            if result.action.value != "allow":
                return result
        return HookResult()
