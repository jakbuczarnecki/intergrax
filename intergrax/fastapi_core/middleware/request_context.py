# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.protocol import ApiHeaders
from intergrax.logging import IntergraxLogging


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware responsible for building and attaching RequestContext
    to every incoming HTTP request.

    Responsibilities:
    - Ensure request_id is present (generate if missing).
    - Build RequestContext from request metadata.
    - Attach context to request.state.context.
    - Propagate X-Request-ID to the response.

    Notes:
    - Auth is resolved here via the configured AuthProvider.
    - No request logging here beyond setting IntergraxLogging context.
    """

    REQUEST_ID_HEADER = ApiHeaders.REQUEST_ID

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id: str = self._get_or_create_request_id(request)

        auth_provider: AuthProvider = request.app.state.auth_provider
        auth = auth_provider.authenticate(request)

        context = RequestContext(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            tenant_id=auth.tenant_id if auth else None,
            user_id=auth.user_id if auth else None,
            auth=auth,
        )

        # Attach context to request state (single source of truth)
        request.state.context = context

        IntergraxLogging.set_context(
            component="api",
            run_id=context.request_id,
            tenant_id=context.tenant_id,
            user_id=context.user_id,
        )

        response: Response = await call_next(request)

        # Propagate request_id to response headers
        response.headers[self.REQUEST_ID_HEADER] = request_id

        return response

    def _get_or_create_request_id(self, request: Request) -> str:
        header_value: Optional[str] = request.headers.get(self.REQUEST_ID_HEADER)
        if header_value:
            return header_value

        return str(uuid.uuid4())
