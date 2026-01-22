# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from fastapi import Depends, HTTPException, Request, status
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.context import update_request_context

@dataclass(frozen=True)
class ApiKeyIdentity:
    """
    Identity resolved from an API key.
    """
    tenant_id: str
    user_id: Optional[str]
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class ApiKeyConfig:
    """
    Static API key configuration.

    This is an MVP in-memory config.
    Will be replaced by DB / secrets manager later.
    """
    keys: Mapping[str, ApiKeyIdentity]


class ApiKeyAuthenticator:
    """
    Resolve AuthContext from X-API-Key header.
    """

    HEADER_NAME: str = "X-API-Key"

    def __init__(self, config: ApiKeyConfig) -> None:
        self._config = config

    def authenticate(self, api_key: str) -> Optional[ApiKeyIdentity]:
        return self._config.keys.get(api_key)


def get_auth_context_from_api_key(
    request: Request,
    authenticator: ApiKeyAuthenticator = Depends(),
) -> AuthContext:
    """
    Resolve AuthContext using API key authentication.
    """
    api_key = request.headers.get(ApiKeyAuthenticator.HEADER_NAME)

    if not api_key:
        return AuthContext(
            is_authenticated=False,
            tenant_id=None,
            user_id=None,
            scopes=(),
        )

    identity = authenticator.authenticate(api_key)
    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    auth = AuthContext(
        is_authenticated=True,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        scopes=identity.scopes,
    )

    # Wire into RequestContext
    update_request_context(
        request,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
    )

    return auth
