# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Iterable

from fastapi import Depends, HTTPException, Request, status

from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.context import update_request_context


def get_auth_context(request: Request) -> AuthContext:
    """
    Resolve authentication context for the current request.

    Skeleton implementation:
    - No token validation.
    - Identity is not authenticated.
    - Wiring to RequestContext is performed.
    """
    auth = AuthContext(
        is_authenticated=False,
        tenant_id=None,
        user_id=None,
        scopes=(),
    )

    # Wire identity into RequestContext (even if None)
    update_request_context(
        request,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
    )

    return auth



def require_auth(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """
    Dependency enforcing authentication.

    Use this on endpoints that require an authenticated identity.
    """
    if not auth.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return auth


def require_scope(required_scope: str):
    """
    Dependency enforcing presence of a specific scope.

    Usage:
        Depends(require_scope("runs:create"))
    """

    def _dependency(auth: AuthContext = Depends(require_auth)) -> AuthContext:
        if require_scope not in auth.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {require_scope}"
            )
        return auth
    
    return _dependency


def require_any_scope(required_scopes: Iterable[str]):
    """
    Dependency enforcing presence of at least one required scope.

    Usage:
        Depends(require_any_scope(["runs:read", "runs:write"]))
    """

    required = set(required_scopes)

    def _dependency(auth: AuthContext = Depends(require_auth)) -> AuthContext:
        if not required.intersection(auth.scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope (any of): {sorted(required)}",
            )
        return auth

    return _dependency

