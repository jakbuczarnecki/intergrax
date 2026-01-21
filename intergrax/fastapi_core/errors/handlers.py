# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from intergrax.fastapi_core.context import get_request_context
from intergrax.fastapi_core.errors.mapping import map_exception_to_api_error
from intergrax.fastapi_core.errors.models import ErrorResponse


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for FastAPI Core.

    Guarantees:
    - Stable ErrorResponse schema.
    - No stack trace leakage to clients.
    - request_id always present.
    """
    # Try to extract request context; fall back gracefully if missing
    try:
        ctx = get_request_context(request)
        request_id: str = ctx.request_id
    except Exception:
        request_id = "unknown"

    error_code, status_code, message = map_exception_to_api_error(exc)

    payload = ErrorResponse(
        error_type=error_code.value,
        message=message,
        request_id=request_id,
    )

    # NOTE: logging will be added in the next step (structured logging)
    return JSONResponse(status_code=status_code, content=payload.__dict__)
