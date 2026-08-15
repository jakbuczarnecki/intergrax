# © Artur Czarnecki. All rights reserved.

"""Optional API-key guard for lab harness HTTP/MCP surfaces (Phase U-Sec.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend


@dataclass(frozen=True, slots=True)
class HarnessAuthState:
    """Typed FastAPI app state for harness authentication."""

    identity_provider: IdentityProviderBackend | None = None
    require_api_key: bool = False


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


def _harness_auth_state_from_request(request: Request) -> HarnessAuthState | None:
    try:
        state = request.app.state.harness_auth
    except AttributeError:
        return None
    if isinstance(state, HarnessAuthState):
        return state
    return None


def _identity_provider_from_request(request: Request) -> IdentityProviderBackend | None:
    state = _harness_auth_state_from_request(request)
    if state is not None and state.identity_provider is not None:
        return state.identity_provider
    return None


def is_harness_identity_token_valid(
    *,
    authorization: str | None,
    identity_provider: IdentityProviderBackend,
) -> bool:
    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    token = authorization[7:].strip()
    if not token:
        return False
    try:
        user = identity_provider.verify_token(token)
    except Exception:  # noqa: BLE001 — auth boundary
        return False
    return bool(user.user_id)


def require_harness_auth(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Enforce static API key and/or OIDC bearer token when configured."""
    if is_harness_api_key_valid(x_api_key=x_api_key, authorization=authorization):
        return
    identity_provider = _identity_provider_from_request(request)
    if identity_provider is not None and is_harness_identity_token_valid(
        authorization=authorization,
        identity_provider=identity_provider,
    ):
        return
    if resolve_harness_api_key() is None and identity_provider is None:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing harness credentials",
    )


def require_agent_platform_admin_auth(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """
    Fail-closed admin auth for Agent Platform control-plane routes (AP-11).

    Allows valid harness API key or configured identity-provider bearer tokens.
    When no credentials are configured, rejects unless the host explicitly wired
    ``HarnessAuthState(require_api_key=False)`` for local development profiles.
    """
    expected_api_key = resolve_harness_api_key()
    if expected_api_key is not None and is_harness_api_key_valid(
        x_api_key=x_api_key,
        authorization=authorization,
    ):
        return
    identity_provider = _identity_provider_from_request(request)
    if identity_provider is not None and is_harness_identity_token_valid(
        authorization=authorization,
        identity_provider=identity_provider,
    ):
        return
    state = _harness_auth_state_from_request(request)
    if (
        state is not None
        and not state.require_api_key
        and expected_api_key is None
        and state.identity_provider is None
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing agent platform admin credentials",
    )


_HARNESS_AUTH_EXEMPT_PATHS = frozenset(
    {"/health", "/openapi.json", "/docs", "/redoc", "/favicon.ico"},
)


class HarnessApiKeyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for lab/MCP wrapper apps (covers mounted sub-apps)."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _HARNESS_AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if is_harness_api_key_valid(
            x_api_key=request.headers.get("X-Api-Key"),
            authorization=request.headers.get("Authorization"),
        ):
            return await call_next(request)
        identity_provider = _identity_provider_from_request(request)
        if identity_provider is not None and is_harness_identity_token_valid(
            authorization=request.headers.get("Authorization"),
            identity_provider=identity_provider,
        ):
            return await call_next(request)
        if resolve_harness_api_key() is None and identity_provider is None:
            return await call_next(request)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing harness credentials"},
        )


def apply_harness_auth_middleware(app, *, require_auth: bool = False) -> None:
    """Attach harness auth middleware when the host profile requires authenticated access."""
    if not require_auth:
        return
    try:
        state = app.state.harness_auth
    except AttributeError:
        state = None
    identity_configured = isinstance(state, HarnessAuthState) and state.identity_provider is not None
    if resolve_harness_api_key() is not None or identity_configured:
        app.add_middleware(HarnessApiKeyMiddleware)
