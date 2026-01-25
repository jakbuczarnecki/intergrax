# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations

from fastapi import Depends
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.fastapi_core.errors.auth import MissingScopeError, NotAuthenticatedError


class AuthRequired:
    """
    FastAPI dependency enforcing authentication.

    This dependency performs NO authentication.
    It only validates that the request has been authenticated
    by RequestContextMiddleware.
    """

    def __call__(
        self,
        context: RequestContext = Depends(get_request_context),
    ) -> None:
        auth = context.auth

        if auth is None or not auth.is_authenticated:
            raise NotAuthenticatedError()



class ScopeRequired:
    """
    FastAPI dependency enforcing presence of a specific authorization scope.
    """

    def __init__(self, required_scope: str) -> None:
        if not required_scope:
            raise ValueError("required_scope must be a non-empty string")

        self._required_scope: str = required_scope

    def __call__(
        self,
        context: RequestContext = Depends(get_request_context),
    ) -> None:
        auth = context.auth

        # Not authenticated at all
        if auth is None or not auth.is_authenticated:
            raise NotAuthenticatedError()

        # Scope missing
        if self._required_scope not in auth.scopes:
            raise MissingScopeError(self._required_scope)