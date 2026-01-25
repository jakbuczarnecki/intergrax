# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from __future__ import annotations

from fastapi import Depends
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.fastapi_core.errors.auth import NotAuthenticatedError


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

        if not auth.is_authenticated:
            raise NotAuthenticatedError()
