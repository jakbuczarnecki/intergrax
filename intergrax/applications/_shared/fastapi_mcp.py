# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Couple FastMCP with an existing FastAPI application (Phase N).

FastMCP session management requires its ASGI lifespan. Tier-3 hosts mount MCP at a
dedicated path (default ``/mcp``) on a wrapper app that also serves the original
FastAPI routes at ``/``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP

from intergrax.applications._shared.harness_auth import HarnessAuthState, apply_harness_auth_middleware
from fastmcp.utilities.lifespan import combine_lifespans

LifespanFn = Callable[[FastAPI], AsyncIterator[None]]


def _normalize_mount_path(mount_path: str) -> str:
    path = (mount_path or "/mcp").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/") or "/mcp"


def make_scheduler_lifespan(scheduler: Any) -> LifespanFn:
    """Lifespan that starts/stops a long-running scheduler (replaces ``on_event``)."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    return _lifespan


def apply_lifespans(app: FastAPI, *lifespans: LifespanFn) -> FastAPI:
    """Merge lifespan handlers onto an existing FastAPI app (replaces ``on_event``)."""
    if not lifespans:
        return app
    existing = app.router.lifespan_context
    app.router.lifespan_context = combine_lifespans(*lifespans, existing)
    return app


def couple_fastapi_with_mcp(
    fastapi_app: FastAPI,
    mcp: FastMCP,
    *,
    mount_path: str = "/mcp",
    extra_lifespans: list[LifespanFn] | None = None,
) -> FastAPI:
    """
    Mount *mcp* on a wrapper FastAPI app that also serves *fastapi_app* at ``/``.

    Returns a new FastAPI instance suitable as the uvicorn entrypoint
    (``host.main:app``). MCP streamable HTTP endpoint: ``{mount_path}`` (with
    ``http_app(path="/")`` — no double prefix).
    """
    mcp_mount = _normalize_mount_path(mount_path)
    mcp_app = mcp.http_app(path="/")

    lifespans: list[LifespanFn] = list(extra_lifespans or [])
    lifespans.append(mcp_app.lifespan)
    combined = combine_lifespans(*lifespans) if len(lifespans) > 1 else mcp_app.lifespan

    wrapper = FastAPI(
        title=fastapi_app.title,
        description=fastapi_app.description,
        version=fastapi_app.version,
        lifespan=combined,
    )
    wrapper.mount(mcp_mount, mcp_app)
    wrapper.mount("/", fastapi_app)
    if hasattr(fastapi_app.state, "harness_auth"):
        inner_auth = fastapi_app.state.harness_auth
        if isinstance(inner_auth, HarnessAuthState):
            wrapper.state.harness_auth = inner_auth
    apply_harness_auth_middleware(wrapper)
    return wrapper
