# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe domain errors for the Vendor Knowledge Facade contract layer."""

from __future__ import annotations

from enum import StrEnum


class VendorKnowledgeErrorCode(StrEnum):
    """Normalized facade error codes — safe for logs and application status."""

    INVALID_REQUEST = "invalid_request"
    CONFIGURATION = "configuration"
    UNKNOWN_SOURCE = "unknown_source"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CURSOR_INVALID = "cursor_invalid"
    ITEM_NOT_FOUND = "item_not_found"
    ITEM_REVOKED = "item_revoked"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"


_DEFAULT_RETRYABLE_CODES: frozenset[VendorKnowledgeErrorCode] = frozenset(
    {
        VendorKnowledgeErrorCode.RATE_LIMIT,
        VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
    }
)

_DEFAULT_NON_RETRYABLE_CODES: frozenset[VendorKnowledgeErrorCode] = frozenset(
    {
        VendorKnowledgeErrorCode.AUTHENTICATION,
        VendorKnowledgeErrorCode.AUTHORIZATION,
        VendorKnowledgeErrorCode.CONFIGURATION,
        VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
    }
)


def _default_retryable(code: VendorKnowledgeErrorCode) -> bool:
    if code in _DEFAULT_RETRYABLE_CODES:
        return True
    if code in _DEFAULT_NON_RETRYABLE_CODES:
        return False
    return False


class VendorKnowledgeError(Exception):
    """Safe domain exception for vendor-knowledge facade boundaries.

    Does not store raw provider response bodies or secret material.
    """

    def __init__(
        self,
        *,
        code: VendorKnowledgeErrorCode,
        safe_message: str,
        provider_id: str | None = None,
        source_kind: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        message = str(safe_message).strip()
        if not message:
            raise ValueError("safe_message must be a non-empty string")
        self.code = code
        self.safe_message = message
        self.provider_id = provider_id
        self.source_kind = source_kind
        self.retryable = _default_retryable(code) if retryable is None else bool(retryable)
        super().__init__(message)

    def __str__(self) -> str:
        return self.safe_message

    def __repr__(self) -> str:
        return (
            "VendorKnowledgeError("
            f"code={self.code!r}, "
            f"safe_message={self.safe_message!r}, "
            f"provider_id={self.provider_id!r}, "
            f"source_kind={self.source_kind!r}, "
            f"retryable={self.retryable!r})"
        )
