# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime middleware base (architecture §42.20)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint


class RuntimeMiddleware(ABC):
    """Base class for middleware registered into the pipeline."""

    priority: int = 100
    name: str = "RuntimeMiddleware"

    @abstractmethod
    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()

    @abstractmethod
    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()
