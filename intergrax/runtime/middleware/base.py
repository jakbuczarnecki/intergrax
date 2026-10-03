# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime middleware base (architecture §42.20)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationMiddlewareHookContext,
)
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.runtime.hooks.hook_context import HookResult


class RuntimeMiddleware(ABC):
    """Base class for middleware registered into the pipeline."""

    priority: int = 100
    name: str = "RuntimeMiddleware"

    @abstractmethod
    async def before(
        self,
        point: HookPoint,
        ctx: HostOrchestrationMiddlewareHookContext,
    ) -> HookResult:
        return HookResult()

    @abstractmethod
    async def after(
        self,
        point: HookPoint,
        ctx: HostOrchestrationMiddlewareHookContext,
    ) -> HookResult:
        return HookResult()
