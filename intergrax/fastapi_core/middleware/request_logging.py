# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from intergrax.fastapi_core.context import get_request_context
from intergrax.logging import IntergraxLogging


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware responsible for HTTP request/response logging
    using the central Intergrax logging mechanism.

    Guarantees:
    - Single logging system across API and runtime.
    - Automatic correlation via IntergraxLogging context.
    - No payload logging (metadata only).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time: float = time.time()

        response: Response
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms: int = int((time.time() - start_time) * 1000)

            try:
                ctx = get_request_context(request)
                logger = IntergraxLogging.get_logger(
                    __name__,
                    component="api",
                )

                logger.info(
                    "http_request",
                    extra={
                        "data": {
                            "request_id": ctx.request_id,
                            "path": ctx.path,
                            "method": ctx.method,
                            "status_code": response.status_code,
                            "latency_ms": elapsed_ms,
                        }
                    },
                )
            except Exception:
                pass
