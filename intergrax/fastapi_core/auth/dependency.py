# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

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
