# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request


@dataclass(frozen=True)
class RequestContext:
    """
    Immutable request-scoped context for FastAPI services.

    This is an infrastructure-level contract used for:
    - observability (correlation, logging),
    - error handling,
    - multi-tenant and user scoping,
    - future auth, rate limiting, and runtime integration.

    Notes:
    - This is NOT an API payload model.
    - No validation logic here; values are populated by middleware.
    """
    request_id: str
    path: str
    method: str
    tenant_id: Optional[str]
    user_id: Optional[str]


def get_request_context(request: Request) -> RequestContext:
    """
    Retrieve RequestContext from FastAPI request state.

    This accessor is intentionally strict:
    - If the context is missing, it raises a RuntimeError,
      because this indicates a misconfigured application
      (middleware not installed or executed).

    Rationale:
    - Avoid silent fallbacks.
    - Fail fast on infrastructure misconfiguration.
    """
    try:
        context = request.state.context
    except AttributeError as exc:
        raise RuntimeError(
            "RequestContext is not available on request.state. "
            "Ensure RequestContextMiddleware is installed and executed."
        ) from exc

    if not isinstance(context, RequestContext):
        raise RuntimeError(
            "Invalid type stored in request.state.context. "
            "Expected RequestContext."
        )

    return context

