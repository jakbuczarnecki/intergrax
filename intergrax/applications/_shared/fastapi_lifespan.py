# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""FastAPI lifespan helpers shared by application hosts.

This module intentionally has no FastMCP dependency. HTTP-only application hosts must be
able to wire cleanup and scheduler lifespans without importing MCP server support.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI

LifespanFn = Callable[[FastAPI], AsyncIterator[None]]


def combine_lifespans(*lifespans: LifespanFn) -> LifespanFn:
    """Combine multiple FastAPI lifespan handlers in declaration order."""

    @asynccontextmanager
    async def _combined(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for lifespan in lifespans:
                await stack.enter_async_context(lifespan(app))
            yield

    return _combined


def make_scheduler_lifespan(scheduler: Any) -> LifespanFn:
    """Lifespan that starts/stops a long-running scheduler."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    return _lifespan


def apply_lifespans(app: FastAPI, *lifespans: LifespanFn) -> FastAPI:
    """Merge lifespan handlers onto an existing FastAPI app."""
    if not lifespans:
        return app
    existing = app.router.lifespan_context
    app.router.lifespan_context = combine_lifespans(*lifespans, existing)
    return app
