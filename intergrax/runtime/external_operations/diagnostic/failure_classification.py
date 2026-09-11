# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map provider exceptions to typed external operation failures (R2)."""

from __future__ import annotations

import asyncio

from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind

_TIMEOUT_TYPES = frozenset({"TimeoutError", "asyncio.TimeoutError"})
_AUTH_TYPES = frozenset({"AuthenticationError", "PermissionError"})
_AUTHZ_TYPES = frozenset({"AuthorizationError", "ForbiddenError"})
_RATE_TYPES = frozenset({"RateLimitError", "TooManyRequests"})
_UNAVAILABLE_TYPES = frozenset({"ConnectionError", "ServiceUnavailable", "ProviderUnavailable"})
_INVALID_TYPES = frozenset({"ValueError", "TypeError", "InvalidRequest", "BadRequest"})
_REMOTE_SUFFIX = "Exception"


def failure_kind_retryable_default(kind: ExternalOperationFailureKind) -> bool:
    return kind in {
        ExternalOperationFailureKind.TIMEOUT,
        ExternalOperationFailureKind.RATE_LIMITED,
        ExternalOperationFailureKind.PROVIDER_UNAVAILABLE,
        ExternalOperationFailureKind.REMOTE_FAILURE,
    }


def classify_provider_exception(exc: BaseException) -> ExternalOperationFailureKind:
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or name in _TIMEOUT_TYPES:
        return ExternalOperationFailureKind.TIMEOUT
    if name in _AUTH_TYPES:
        return ExternalOperationFailureKind.AUTHENTICATION_FAILED
    if name in _AUTHZ_TYPES:
        return ExternalOperationFailureKind.AUTHORIZATION_FAILED
    if name in _RATE_TYPES:
        return ExternalOperationFailureKind.RATE_LIMITED
    if name in _INVALID_TYPES:
        return ExternalOperationFailureKind.INVALID_REQUEST
    if name in _UNAVAILABLE_TYPES:
        return ExternalOperationFailureKind.PROVIDER_UNAVAILABLE
    if name.endswith(_REMOTE_SUFFIX) or name in {"SAPException", "CRMProviderError"}:
        return ExternalOperationFailureKind.REMOTE_FAILURE
    return ExternalOperationFailureKind.UNKNOWN


__all__ = ["classify_provider_exception", "failure_kind_retryable_default"]
