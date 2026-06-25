# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Couple FastMCP with an existing FastAPI application (Phase N).

FastMCP session management requires its ASGI lifespan. Tier-3 hosts mount MCP at a
dedicated path (default ``/mcp``) on a wrapper app that also serves the original
FastAPI routes at ``/``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastmcp import FastMCP

from intergrax.applications._shared.fastapi_lifespan import LifespanFn, combine_lifespans
from intergrax.applications._shared.harness_auth import HarnessAuthState, apply_harness_auth_middleware


def _normalize_mount_path(mount_path: str) -> str:
    path = (mount_path or "/mcp").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/") or "/mcp"


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
    require_auth = False
    if hasattr(fastapi_app.state, "harness_auth"):
        inner_auth = fastapi_app.state.harness_auth
        if isinstance(inner_auth, HarnessAuthState):
            wrapper.state.harness_auth = inner_auth
            require_auth = inner_auth.require_api_key
    apply_harness_auth_middleware(wrapper, require_auth=require_auth)
    return wrapper
