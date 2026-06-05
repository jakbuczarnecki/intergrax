# © Artur Czarnecki. All rights reserved.

"""Inject trace/run correlation into request scope (Phase DX-5.6)."""

from __future__ import annotations

import logging
import uuid
from typing import Callable

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("intergrax.harness")


class HarnessCorrelationMiddleware(BaseHTTPMiddleware):
    """Adds trace_id and run_id headers; binds them to log context."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        trace_id = request.headers.get("x-trace-id", uuid.uuid4().hex)
        run_id = request.headers.get("x-run-id", uuid.uuid4().hex)
        request.state.trace_id = trace_id
        request.state.run_id = run_id
        logger.info("request_start path=%s trace_id=%s run_id=%s", request.url.path, trace_id, run_id)
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        response.headers["x-run-id"] = run_id
        return response


def apply_harness_correlation_middleware(app: FastAPI) -> None:
    app.add_middleware(HarnessCorrelationMiddleware)
