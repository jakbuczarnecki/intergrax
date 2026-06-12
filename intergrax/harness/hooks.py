# © Artur Czarnecki. All rights reserved.

"""Bridge :class:`~intergrax.harness.application_host.ApplicationHost` to middleware (Phase DX-5.2)."""

from __future__ import annotations

from intergrax.harness.application_host import ApplicationHost
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline


class ApplicationHostMiddleware(RuntimeMiddleware):
    """Invokes a typed :class:`ApplicationHost` at Nexus hook points."""

    priority = 50
    name = "application_host"

    def __init__(self, host: ApplicationHost) -> None:
        self._host = host

    async def before(self, point: HookPoint, context: HookContext) -> HookResult:
        result = self._host.on_hook(point, context)
        if result is not None:
            return result
        return HookResult()

    async def after(self, point: HookPoint, context: HookContext) -> HookResult:
        return HookResult()


def application_host_middleware(host: ApplicationHost | None) -> list[RuntimeMiddleware]:
    if host is None:
        return []
    return [ApplicationHostMiddleware(host)]


def merge_host_into_pipeline(
    pipeline: MiddlewarePipeline,
    host: ApplicationHost | None,
) -> MiddlewarePipeline:
    """Return a pipeline that includes the application host middleware."""
    extra = application_host_middleware(host)
    if not extra:
        return pipeline
    combined = list(pipeline._middleware) + extra  # noqa: SLF001 — harness composition
    merged = MiddlewarePipeline(hook_registry=pipeline.hooks, middleware=combined)
    merged.configure_hook_runtime(
        hook_timeout_seconds=pipeline._hook_timeout_seconds,  # noqa: SLF001
        event_bus=pipeline._event_bus,  # noqa: SLF001
    )
    return merged
