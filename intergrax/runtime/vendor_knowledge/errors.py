# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe domain errors for the Vendor Knowledge Facade contract layer."""

from __future__ import annotations

import re
from enum import StrEnum
from urllib.parse import parse_qsl, urlparse


class VendorKnowledgeErrorCode(StrEnum):
    """Normalized facade error codes — safe for logs and application status."""

    CONFIGURATION_ERROR = "configuration_error"
    INTEGRATION_NOT_FOUND = "integration_not_found"
    INTEGRATION_CATEGORY_MISMATCH = "integration_category_mismatch"
    ADAPTER_NOT_FOUND = "adapter_not_found"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_DENIED = "authorization_denied"
    RATE_LIMITED = "rate_limited"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    REMOTE_ITEM_NOT_FOUND = "remote_item_not_found"
    REMOTE_ITEM_REVOKED = "remote_item_revoked"
    INVALID_CURSOR = "invalid_cursor"
    INVALID_SCOPE = "invalid_scope"
    TENANT_MISMATCH = "tenant_mismatch"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"


_DEFAULT_RETRYABLE_CODES: frozenset[VendorKnowledgeErrorCode] = frozenset(
    {
        VendorKnowledgeErrorCode.RATE_LIMITED,
        VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
    }
)

_URL_IN_TEXT_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_AUTHORIZATION_ASSIGN_RE = re.compile(r"\bauthorization\s*[:=]", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._\-+=/]+", re.IGNORECASE)
_API_KEY_ASSIGN_RE = re.compile(r"\bapi[_-]?key\s*[:=]", re.IGNORECASE)

_SECRET_QUERY_NAMES: frozenset[str] = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "api_key",
        "authorization",
        "credential",
        "bearer",
    }
)


def _default_retryable(code: VendorKnowledgeErrorCode) -> bool:
    return code in _DEFAULT_RETRYABLE_CODES


def _is_forbidden_query_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if not normalized:
        return False
    if normalized in _SECRET_QUERY_NAMES:
        return True
    return any(
        normalized.endswith(suffix)
        for suffix in (
            "_token",
            "_password",
            "_secret",
            "_api_key",
            "_authorization",
            "_credential",
            "_bearer",
        )
    )


def _url_embeds_secrets(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.username is not None or parsed.password is not None:
        return True
    for raw_key, _raw_value in parse_qsl(parsed.query, keep_blank_values=True):
        if _is_forbidden_query_key(raw_key):
            return True
    return False


def _assert_safe_public_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if _AUTHORIZATION_ASSIGN_RE.search(cleaned):
        raise ValueError(f"{field_name} must not contain authorization header material")
    if _BEARER_TOKEN_RE.search(cleaned):
        raise ValueError(f"{field_name} must not contain bearer token material")
    if _API_KEY_ASSIGN_RE.search(cleaned):
        raise ValueError(f"{field_name} must not contain api-key material")
    for match in _URL_IN_TEXT_RE.finditer(cleaned):
        candidate = match.group(0).rstrip(".,);]'\"")
        if _url_embeds_secrets(candidate):
            raise ValueError(
                f"{field_name} must not contain credential-bearing or secret-bearing URLs"
            )
    return cleaned


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
        message = _assert_safe_public_text(str(safe_message), field_name="safe_message")
        resolved_provider: str | None = None
        if provider_id is not None:
            resolved_provider = _assert_safe_public_text(
                str(provider_id), field_name="provider_id"
            )
        resolved_source: str | None = None
        if source_kind is not None:
            resolved_source = _assert_safe_public_text(
                str(source_kind), field_name="source_kind"
            )
        self.code = code
        self.safe_message = message
        self.provider_id = resolved_provider
        self.source_kind = resolved_source
        self.retryable = _default_retryable(code) if retryable is None else bool(retryable)
        super().__init__(message)

    def __str__(self) -> str:
        return self.safe_message

    def __repr__(self) -> str:
        return (
            "VendorKnowledgeError("
            f"code={self.code!r}, "
            f"provider_id={self.provider_id!r}, "
            f"source_kind={self.source_kind!r}, "
            f"retryable={self.retryable!r})"
        )
