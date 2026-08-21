# © Artur Czarnecki. All rights reserved.

"""Optional API-key guard for lab harness HTTP/MCP surfaces (Phase U-Sec.1)."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from intergrax.integrations.contracts.identity_provider import (
    IdentityProviderBackend,
    IdentityUser,
    identity_user_has_agent_platform_admin_authority,
)


@dataclass(frozen=True, slots=True)
class HarnessAuthState:
    """Typed FastAPI app state for harness authentication."""

    identity_provider: IdentityProviderBackend | None = None
    require_api_key: bool = False
    resolved_api_key: str | None = None


def _legacy_default_harness_api_key() -> str | None:
    """Resolve the default harness env key for unwired legacy hosts."""
    raw = (os.getenv("INTERGRAX_HARNESS_API_KEY") or "").strip()
    return raw or None


def resolve_harness_api_key() -> str | None:
    """Return the default harness env key for unwired legacy hosts."""
    return _legacy_default_harness_api_key()


def _expected_api_key_for_request(request: Request | None) -> str | None:
    if request is None:
        return _legacy_default_harness_api_key()
    state = _harness_auth_state_from_request(request)
    if state is not None:
        return state.resolved_api_key
    return _legacy_default_harness_api_key()


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
    expected_api_key: str | None = None,
    request: Request | None = None,
) -> bool:
    """Return whether headers satisfy the effective configured API key authority."""
    expected = expected_api_key
    if expected is None:
        expected = _expected_api_key_for_request(request)
    if expected is None:
        return True
    provided = _extract_provided_key(x_api_key=x_api_key, authorization=authorization)
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def require_harness_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """
    Enforce the effective harness API key when configured.

    Accepts ``X-Api-Key`` or ``Authorization: Bearer <key>``.
    When no key authority is configured, all requests pass (local dev default).
    """
    expected = _expected_api_key_for_request(request)
    if expected is None:
        return

    if not is_harness_api_key_valid(
        x_api_key=x_api_key,
        authorization=authorization,
        expected_api_key=expected,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing harness API key",
        )


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


def verify_harness_bearer_identity(
    *,
    authorization: str | None,
    identity_provider: IdentityProviderBackend,
) -> IdentityUser | None:
    """Validate bearer token and return the provider-normalized authenticated principal."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    if not token:
        return None
    try:
        user = identity_provider.verify_token(token)
    except Exception:  # noqa: BLE001 — auth boundary
        return None
    if not user.user_id.strip():
        return None
    return user


def is_harness_identity_token_valid(
    *,
    authorization: str | None,
    identity_provider: IdentityProviderBackend,
) -> bool:
    return verify_harness_bearer_identity(
        authorization=authorization,
        identity_provider=identity_provider,
    ) is not None


def _local_dev_auth_bypass_allowed(state: HarnessAuthState | None) -> bool:
    return (
        state is not None
        and not state.require_api_key
        and state.resolved_api_key is None
        and state.identity_provider is None
    )


def require_harness_auth(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Enforce static API key and/or OIDC bearer token when configured."""
    if is_harness_api_key_valid(
        x_api_key=x_api_key,
        authorization=authorization,
        request=request,
    ):
        return
    identity_provider = _identity_provider_from_request(request)
    if identity_provider is not None and is_harness_identity_token_valid(
        authorization=authorization,
        identity_provider=identity_provider,
    ):
        return
    state = _harness_auth_state_from_request(request)
    if _local_dev_auth_bypass_allowed(state):
        return
    if _expected_api_key_for_request(request) is None and identity_provider is None:
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

    Allows the resolved harness API key or identity-provider bearer tokens that
    carry explicit Agent Platform admin authority. When no credentials are
    configured, rejects unless the host explicitly wired a local development
    profile with ``HarnessAuthState(require_api_key=False)``.
    """
    expected_api_key = _expected_api_key_for_request(request)
    if expected_api_key is not None and is_harness_api_key_valid(
        x_api_key=x_api_key,
        authorization=authorization,
        expected_api_key=expected_api_key,
    ):
        return
    identity_provider = _identity_provider_from_request(request)
    if identity_provider is not None:
        user = verify_harness_bearer_identity(
            authorization=authorization,
            identity_provider=identity_provider,
        )
        if user is not None:
            if identity_user_has_agent_platform_admin_authority(user):
                return
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Agent platform admin authorization required",
            )
    state = _harness_auth_state_from_request(request)
    if _local_dev_auth_bypass_allowed(state):
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
            request=request,
        ):
            return await call_next(request)
        identity_provider = _identity_provider_from_request(request)
        if identity_provider is not None and is_harness_identity_token_valid(
            authorization=request.headers.get("Authorization"),
            identity_provider=identity_provider,
        ):
            return await call_next(request)
        state = _harness_auth_state_from_request(request)
        if _local_dev_auth_bypass_allowed(state):
            return await call_next(request)
        if _expected_api_key_for_request(request) is None and identity_provider is None:
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
    if isinstance(state, HarnessAuthState):
        if state.resolved_api_key is not None or state.identity_provider is not None:
            app.add_middleware(HarnessApiKeyMiddleware)
        return
    if _legacy_default_harness_api_key() is not None:
        app.add_middleware(HarnessApiKeyMiddleware)
