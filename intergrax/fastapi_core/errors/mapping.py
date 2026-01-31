# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Tuple

from fastapi import HTTPException, status

from intergrax.fastapi_core.errors.auth import MissingScopeError, NotAuthenticatedError
from intergrax.fastapi_core.errors.budget import BudgetExceededError
from intergrax.fastapi_core.errors.error_types import ApiErrorType
from intergrax.fastapi_core.rate_limit.errors import RateLimitExceededError


def map_exception_to_api_error(exc: Exception) -> Tuple[ApiErrorType, int, str]:
    """
    Map an exception to:
    - ApiErrorType,
    - HTTP status code,
    - safe, client-facing message.
    """

    if isinstance(exc, BudgetExceededError):
        return ApiErrorType.RATE_LIMITED, status.HTTP_429_TOO_MANY_REQUESTS, ""

    if isinstance(exc, NotAuthenticatedError):
        return ApiErrorType.UNAUTHORIZED, status.HTTP_401_UNAUTHORIZED, ""
    
    if isinstance(exc, MissingScopeError):
        return ApiErrorType.FORBIDDEN, status.HTTP_403_FORBIDDEN, ""
    
    if isinstance(exc, RateLimitExceededError):
        return ApiErrorType.RATE_LIMITED, status.HTTP_429_TOO_MANY_REQUESTS, ""

    if isinstance(exc, HTTPException):
        if exc.status_code == status.HTTP_400_BAD_REQUEST:
            return ApiErrorType.BAD_REQUEST, exc.status_code, exc.detail
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return ApiErrorType.UNAUTHORIZED, exc.status_code, exc.detail
        if exc.status_code == status.HTTP_403_FORBIDDEN:
            return ApiErrorType.FORBIDDEN, exc.status_code, exc.detail
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            return ApiErrorType.NOT_FOUND, exc.status_code, exc.detail

        # Fallback for other HTTP errors
        return ApiErrorType.BAD_REQUEST, exc.status_code, exc.detail

    return ApiErrorType.INTERNAL_ERROR, status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error"
