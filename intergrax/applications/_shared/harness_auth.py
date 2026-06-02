# © Artur Czarnecki. All rights reserved.

"""Optional API-key guard for lab harness HTTP/MCP surfaces (Phase U-Sec.1)."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def resolve_harness_api_key() -> str | None:
    """Return configured key, or ``None`` when harness auth is disabled."""
    raw = (os.getenv("INTERGRAX_HARNESS_API_KEY") or "").strip()
    return raw or None


def require_harness_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """
    Enforce ``INTERGRAX_HARNESS_API_KEY`` when set.

    Accepts ``X-Api-Key`` or ``Authorization: Bearer <key>``.
    When the env var is unset, all requests pass (local dev default).
    """
    expected = resolve_harness_api_key()
    if expected is None:
        return

    if not is_harness_api_key_valid(x_api_key=x_api_key, authorization=authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing harness API key",
        )


def _extract_provided_key(
    *,
    x_api_key: str | None,
    authorization: str | None,
) -> str | None:
    provided = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    return provided


def is_harness_api_key_valid(
    *,
    x_api_key: str | None = None,
    authorization: str | None = None,
) -> bool:
    """Return whether headers satisfy ``INTERGRAX_HARNESS_API_KEY`` when configured."""
    expected = resolve_harness_api_key()
    if expected is None:
        return True
    provided = _extract_provided_key(x_api_key=x_api_key, authorization=authorization)
    return bool(provided) and provided == expected


class HarnessApiKeyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for lab/MCP wrapper apps (covers mounted sub-apps)."""

    async def dispatch(self, request: Request, call_next):
        expected = resolve_harness_api_key()
        if expected is None:
            return await call_next(request)
        if is_harness_api_key_valid(
            x_api_key=request.headers.get("X-Api-Key"),
            authorization=request.headers.get("Authorization"),
        ):
            return await call_next(request)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing harness API key"},
        )


def apply_harness_auth_middleware(app) -> None:
    """Attach API-key middleware when ``INTERGRAX_HARNESS_API_KEY`` is set."""
    if resolve_harness_api_key() is not None:
        app.add_middleware(HarnessApiKeyMiddleware)
