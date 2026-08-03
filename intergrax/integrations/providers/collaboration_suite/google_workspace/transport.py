# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace shared read transport, retry policy and safe error boundary."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import StrEnum
from time import sleep
from typing import Protocol, TypeVar

from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceBinaryPayload,
    GoogleWorkspaceHttpResponse,
    GoogleWorkspaceRequestExecutor,
    GoogleWorkspaceSourceKind,
)

_T = TypeVar("_T")

_MAX_PAGE_TOKEN_LENGTH = 4096
_MAX_SAFE_REASON_LENGTH = 128
_MAX_EXPECTED_CONTENT_TYPE_LENGTH = 255
_ABSOLUTE_BINARY_MAX_BYTES = 104_857_600

_MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$"
)
_CONTENT_RANGE_PATTERN = re.compile(
    r"^bytes 0-(\d+)/(\d+)$"
)

_FORBIDDEN_QUERY_PARAM_NAMES = frozenset(
    {
        "access_token",
        "oauth_token",
        "authorization",
        "api_key",
        "key",
        "client_secret",
        "refresh_token",
    }
)

_FORBIDDEN_HEADER_NAMES = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-goog-api-key",
    }
)

_RATE_LIMIT_REASONS = frozenset(
    {
        "ratelimitexceeded",
        "userratelimitexceeded",
        "quotaexceeded",
        "sharingratelimitexceeded",
    }
)

_SERVICE_ROOTS: dict[GoogleWorkspaceSourceKind, str] = {
    GoogleWorkspaceSourceKind.DRIVE: "https://www.googleapis.com/drive/v3",
    GoogleWorkspaceSourceKind.DOCS: "https://docs.googleapis.com/v1",
    GoogleWorkspaceSourceKind.SHEETS: "https://sheets.googleapis.com/v4",
    GoogleWorkspaceSourceKind.SLIDES: "https://slides.googleapis.com/v1",
    GoogleWorkspaceSourceKind.CALENDAR: "https://www.googleapis.com/calendar/v3",
    GoogleWorkspaceSourceKind.MAIL: "https://gmail.googleapis.com/gmail/v1",
    GoogleWorkspaceSourceKind.CHAT: "https://chat.googleapis.com/v1",
}


class GoogleWorkspaceErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    TEMPORARY = "temporary"
    MALFORMED_RESPONSE = "malformed_response"
    UNEXPECTED_REDIRECT = "unexpected_redirect"
    PAYLOAD_TOO_LARGE = "payload_too_large"


_ERROR_CODE_BY_KIND: dict[GoogleWorkspaceErrorKind, str] = {
    GoogleWorkspaceErrorKind.INVALID_REQUEST: "GOOGLE_WORKSPACE_INVALID_REQUEST",
    GoogleWorkspaceErrorKind.AUTHENTICATION: "GOOGLE_WORKSPACE_AUTHENTICATION_FAILED",
    GoogleWorkspaceErrorKind.AUTHORIZATION: "GOOGLE_WORKSPACE_AUTHORIZATION_FAILED",
    GoogleWorkspaceErrorKind.NOT_FOUND: "GOOGLE_WORKSPACE_NOT_FOUND",
    GoogleWorkspaceErrorKind.RATE_LIMITED: "GOOGLE_WORKSPACE_RATE_LIMITED",
    GoogleWorkspaceErrorKind.TEMPORARY: "GOOGLE_WORKSPACE_TEMPORARY_FAILURE",
    GoogleWorkspaceErrorKind.MALFORMED_RESPONSE: "GOOGLE_WORKSPACE_MALFORMED_RESPONSE",
    GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT: "GOOGLE_WORKSPACE_UNEXPECTED_REDIRECT",
    GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE: "GOOGLE_WORKSPACE_PAYLOAD_TOO_LARGE",
}

_RETRYABLE_KINDS = frozenset(
    {
        GoogleWorkspaceErrorKind.RATE_LIMITED,
        GoogleWorkspaceErrorKind.TEMPORARY,
    }
)


def _require_exact_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an int")
    return value


def _require_finite_number(value: object, field_name: str) -> int | float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite int or float")
    return value


class GoogleWorkspaceApiError(Exception):
    """Safe provider error without response body, URL query or credential material."""

    def __init__(
        self,
        *,
        kind: GoogleWorkspaceErrorKind,
        status_code: int | None,
        retry_after_seconds: float | None,
        safe_reason: str,
        attempts: int,
    ) -> None:
        self.error_code = _ERROR_CODE_BY_KIND[kind]
        self.kind = kind
        self.status_code = status_code
        self.retryable = kind in _RETRYABLE_KINDS
        self.retry_after_seconds = retry_after_seconds
        self.safe_reason = safe_reason
        self.attempts = attempts
        super().__init__(self.error_code)

    def __str__(self) -> str:
        return self.error_code


@dataclass(frozen=True, slots=True)
class GoogleWorkspaceRetryPolicy:
    max_attempts: int = 3
    request_timeout_seconds: float = 30.0
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    max_retry_after_seconds: float = 30.0
    max_response_bytes: int = 4_194_304

    def __post_init__(self) -> None:
        _require_exact_int(self.max_attempts, "max_attempts")
        _require_exact_int(self.max_response_bytes, "max_response_bytes")
        _require_finite_number(self.request_timeout_seconds, "request_timeout_seconds")
        _require_finite_number(self.base_backoff_seconds, "base_backoff_seconds")
        _require_finite_number(self.max_backoff_seconds, "max_backoff_seconds")
        _require_finite_number(self.max_retry_after_seconds, "max_retry_after_seconds")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if not 0 < self.request_timeout_seconds <= 120:
            raise ValueError("request_timeout_seconds must be > 0 and <= 120")
        if not 0 <= self.base_backoff_seconds <= 10:
            raise ValueError("base_backoff_seconds must be >= 0 and <= 10")
        if not self.base_backoff_seconds <= self.max_backoff_seconds <= 60:
            raise ValueError("max_backoff_seconds must be >= base_backoff_seconds and <= 60")
        if not 0 <= self.max_retry_after_seconds <= 120:
            raise ValueError("max_retry_after_seconds must be >= 0 and <= 120")
        if not 1 <= self.max_response_bytes <= 16_777_216:
            raise ValueError("max_response_bytes must be between 1 and 16777216")


@dataclass(frozen=True, slots=True)
class GoogleWorkspacePageToken:
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.value) is not str:
            raise TypeError("page token must be a string")
        cleaned = self.value.strip()
        if not cleaned:
            raise ValueError("page token must not be blank")
        if len(cleaned) > _MAX_PAGE_TOKEN_LENGTH:
            raise ValueError("page token exceeds maximum length")
        if any(ord(ch) < 32 for ch in cleaned):
            raise ValueError("page token contains control characters")


@dataclass(frozen=True, slots=True)
class GoogleWorkspaceCollectionPage:
    items: tuple[dict[str, object], ...]
    next_page_token: GoogleWorkspacePageToken | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise TypeError("items must be a tuple")
        for item in self.items:
            if not isinstance(item, dict):
                raise TypeError("each item must be a dict")


class _JitterSource(Protocol):
    def __call__(self) -> float:
        """Return a jitter factor in the range [0, 1)."""


class _Sleeper(Protocol):
    def __call__(self, seconds: float) -> None:
        """Sleep for the given number of seconds."""


def _default_jitter() -> float:
    return random.random()


def _default_sleeper(seconds: float) -> None:
    sleep(seconds)


def _validate_relative_path(path: object) -> str:
    if not isinstance(path, str):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    cleaned = path.strip()
    if not cleaned:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    if not cleaned.startswith("/") or cleaned.startswith("//"):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    if "://" in cleaned or "@" in cleaned or "?" in cleaned or "#" in cleaned or "\\" in cleaned:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    if any(ord(ch) < 32 for ch in cleaned):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    segments = cleaned.split("/")
    if "." in segments or ".." in segments:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_relative_path",
            attempts=0,
        )
    return cleaned


def _validate_items_field(items_field: object) -> str:
    if not isinstance(items_field, str):
        raise ValueError("items_field must be a string")
    cleaned = items_field.strip()
    if not cleaned:
        raise ValueError("items_field must not be blank")
    if any(ord(ch) < 32 for ch in cleaned):
        raise ValueError("items_field contains control characters")
    return cleaned


def _copy_params(params: Mapping[str, object] | None) -> dict[str, object] | None:
    if params is None:
        return None
    if not isinstance(params, Mapping):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_query_parameter",
            attempts=0,
        )
    try:
        items = tuple(params.items())
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_query_parameter",
            attempts=0,
        ) from None
    copied: dict[str, object] = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="invalid_query_parameter",
                attempts=0,
            )
        name, value = item
        if not isinstance(name, str):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="invalid_query_parameter",
                attempts=0,
            )
        if name.casefold() in _FORBIDDEN_QUERY_PARAM_NAMES:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="forbidden_query_parameter",
                attempts=0,
            )
        copied[name] = value
    return copied


def _copy_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    copied: dict[str, str] = {}
    if headers is not None:
        if not isinstance(headers, Mapping):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="invalid_header",
                attempts=0,
            )
        try:
            items = tuple(headers.items())
        except Exception:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="invalid_header",
                attempts=0,
            ) from None
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise GoogleWorkspaceApiError(
                    kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                    status_code=None,
                    retry_after_seconds=None,
                    safe_reason="invalid_header",
                    attempts=0,
                )
            name, value = item
            if not isinstance(name, str) or not isinstance(value, str):
                raise GoogleWorkspaceApiError(
                    kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                    status_code=None,
                    retry_after_seconds=None,
                    safe_reason="invalid_header",
                    attempts=0,
                )
            folded_name = name.casefold()
            if folded_name == "accept" or folded_name in _FORBIDDEN_HEADER_NAMES:
                raise GoogleWorkspaceApiError(
                    kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
                    status_code=None,
                    retry_after_seconds=None,
                    safe_reason="forbidden_header",
                    attempts=0,
                )
            copied[name] = value
    copied["Accept"] = "application/json"
    return copied


def _build_service_url(
    *,
    source_kind: GoogleWorkspaceSourceKind,
    relative_path: str,
) -> str:
    if not isinstance(source_kind, GoogleWorkspaceSourceKind):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_source_kind",
            attempts=0,
        )
    root = _SERVICE_ROOTS[source_kind]
    return f"{root}{relative_path}"


def _normalize_header_mapping(
    headers: object,
    *,
    status_code: int | None,
    attempts: int,
) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_response_headers",
            attempts=attempts,
        )
    try:
        items = tuple(headers.items())
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_response_headers",
            attempts=attempts,
        ) from None
    normalized: dict[str, str] = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_response_headers",
                attempts=attempts,
            )
        key, value = item
        if not isinstance(key, str) or not isinstance(value, str):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_response_headers",
                attempts=attempts,
            )
        normalized[key] = value
    return normalized


def _validate_response_shape(
    response: GoogleWorkspaceHttpResponse,
    *,
    policy: GoogleWorkspaceRetryPolicy,
    attempts: int,
    success_max_bytes: int,
    binary_success: bool,
) -> tuple[int, dict[str, str], bytes]:
    try:
        status_code = response.status_code
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_status_code",
            attempts=attempts,
        ) from None
    if type(status_code) is not int or status_code < 100 or status_code > 599:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_status_code",
            attempts=attempts,
        )
    try:
        raw_headers = response.headers
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_response_headers",
            attempts=attempts,
        ) from None
    headers = _normalize_header_mapping(raw_headers, status_code=status_code, attempts=attempts)
    try:
        content = response.content
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_response_content",
            attempts=attempts,
        ) from None
    if type(content) is not bytes:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_response_content",
            attempts=attempts,
        )
    if 200 <= status_code <= 299:
        content_limit = success_max_bytes
    else:
        content_limit = policy.max_response_bytes
    if len(content) > content_limit:
        if 200 <= status_code <= 299 and binary_success:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="payload_too_large",
                attempts=attempts,
            )
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="response_too_large",
            attempts=attempts,
        )
    return status_code, headers, content


def _validate_expected_content_type(value: object) -> str:
    if type(value) is not str:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    if not value or value != value.strip():
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    if any(ord(ch) < 32 for ch in value):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    if "," in value or "*" in value or ";" in value:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    if len(value) > _MAX_EXPECTED_CONTENT_TYPE_LENGTH:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    normalized = value.lower()
    if not _MEDIA_TYPE_PATTERN.fullmatch(normalized):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_expected_content_type",
            attempts=0,
        )
    return normalized


def _validate_binary_max_bytes(value: object) -> int:
    if type(value) is not int:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_max_bytes",
            attempts=0,
        )
    if not 1 <= value <= _ABSOLUTE_BINARY_MAX_BYTES:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_max_bytes",
            attempts=0,
        )
    return value


def _validate_range_mode(value: object) -> bool:
    if type(value) is not bool:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.INVALID_REQUEST,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_range_mode",
            attempts=0,
        )
    return value


def _build_binary_request_headers(
    *,
    expected_content_type: str,
    max_bytes: int,
    range_limited: bool,
) -> dict[str, str]:
    headers = {"Accept": expected_content_type}
    if range_limited:
        headers["Range"] = f"bytes=0-{max_bytes}"
    return headers


def _extract_unique_response_header(
    headers: Mapping[str, str],
    canonical_name: str,
    *,
    status_code: int,
    attempts: int,
    safe_reason: str,
) -> str | None:
    matched_key: str | None = None
    matched_value: str | None = None
    folded_target = canonical_name.casefold()
    for key, value in headers.items():
        if key.casefold() != folded_target:
            continue
        if matched_key is not None and matched_key != key:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason=safe_reason,
                attempts=attempts,
            )
        matched_key = key
        matched_value = value
    return matched_value


def _parse_response_media_type(
    raw_value: str,
    *,
    expected_content_type: str,
    status_code: int,
    attempts: int,
) -> str:
    base = raw_value.split(";", 1)[0].strip().lower()
    if not base or not _MEDIA_TYPE_PATTERN.fullmatch(base):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_type",
            attempts=attempts,
        )
    if base != expected_content_type:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_type",
            attempts=attempts,
        )
    return base


def _validate_optional_content_length(
    headers: Mapping[str, str],
    *,
    content: bytes,
    status_code: int,
    attempts: int,
) -> None:
    raw_value = _extract_unique_response_header(
        headers,
        "Content-Length",
        status_code=status_code,
        attempts=attempts,
        safe_reason="invalid_content_length",
    )
    if raw_value is None:
        return
    if not raw_value or any(ch.isspace() for ch in raw_value):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_length",
            attempts=attempts,
        )
    if raw_value[0] in {"+", "-"} or not raw_value.isdigit():
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_length",
            attempts=attempts,
        )
    parsed = int(raw_value)
    if parsed != len(content):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_length",
            attempts=attempts,
        )


def _decode_success_binary(
    *,
    status_code: int,
    headers: Mapping[str, str],
    content: bytes,
    attempts: int,
    expected_content_type: str,
    max_bytes: int,
    range_limited: bool,
) -> GoogleWorkspaceBinaryPayload:
    raw_content_type = _extract_unique_response_header(
        headers,
        "Content-Type",
        status_code=status_code,
        attempts=attempts,
        safe_reason="invalid_content_type",
    )
    if raw_content_type is None:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_content_type",
            attempts=attempts,
        )
    content_type = _parse_response_media_type(
        raw_content_type,
        expected_content_type=expected_content_type,
        status_code=status_code,
        attempts=attempts,
    )
    _validate_optional_content_length(
        headers,
        content=content,
        status_code=status_code,
        attempts=attempts,
    )

    if not range_limited:
        if status_code == 206:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="unexpected_partial_content",
                attempts=attempts,
            )
        if status_code != 200:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="unexpected_binary_status",
                attempts=attempts,
            )
        if len(content) > max_bytes:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="payload_too_large",
                attempts=attempts,
            )
        return GoogleWorkspaceBinaryPayload(data=content, content_type=content_type)

    if status_code == 200:
        if len(content) > max_bytes:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="payload_too_large",
                attempts=attempts,
            )
        return GoogleWorkspaceBinaryPayload(data=content, content_type=content_type)

    if status_code == 206:
        raw_range = _extract_unique_response_header(
            headers,
            "Content-Range",
            status_code=status_code,
            attempts=attempts,
            safe_reason="invalid_content_range",
        )
        if raw_range is None:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_content_range",
                attempts=attempts,
            )
        match = _CONTENT_RANGE_PATTERN.fullmatch(raw_range.strip())
        if match is None:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_content_range",
                attempts=attempts,
            )
        end = int(match.group(1))
        total = int(match.group(2))
        if end + 1 != len(content):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_content_range",
                attempts=attempts,
            )
        if total < len(content):
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="invalid_content_range",
                attempts=attempts,
            )
        if total > max_bytes or len(content) > max_bytes:
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.PAYLOAD_TOO_LARGE,
                status_code=status_code,
                retry_after_seconds=None,
                safe_reason="payload_too_large",
                attempts=attempts,
            )
        return GoogleWorkspaceBinaryPayload(data=content, content_type=content_type)

    raise GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
        status_code=status_code,
        retry_after_seconds=None,
        safe_reason="unexpected_binary_status",
        attempts=attempts,
    )


def _extract_safe_reason(content: bytes) -> str:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    reason = ""
    errors = error.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            candidate = first.get("reason")
            if isinstance(candidate, str):
                reason = candidate
    if not reason:
        status = error.get("status")
        if isinstance(status, str):
            reason = status
    cleaned = reason.strip()
    if not cleaned:
        return ""
    if len(cleaned) > _MAX_SAFE_REASON_LENGTH:
        cleaned = cleaned[:_MAX_SAFE_REASON_LENGTH]
    if any(ord(ch) < 32 for ch in cleaned):
        return ""
    return cleaned


def _classify_status(
    status_code: int,
    *,
    safe_reason: str,
) -> GoogleWorkspaceErrorKind:
    if 300 <= status_code <= 399:
        return GoogleWorkspaceErrorKind.UNEXPECTED_REDIRECT
    if status_code == 400:
        return GoogleWorkspaceErrorKind.INVALID_REQUEST
    if status_code == 401:
        return GoogleWorkspaceErrorKind.AUTHENTICATION
    if status_code == 403:
        if safe_reason.casefold() in _RATE_LIMIT_REASONS:
            return GoogleWorkspaceErrorKind.RATE_LIMITED
        return GoogleWorkspaceErrorKind.AUTHORIZATION
    if status_code == 404:
        return GoogleWorkspaceErrorKind.NOT_FOUND
    if status_code == 408:
        return GoogleWorkspaceErrorKind.TEMPORARY
    if status_code == 429:
        return GoogleWorkspaceErrorKind.RATE_LIMITED
    if status_code >= 500:
        return GoogleWorkspaceErrorKind.TEMPORARY
    if 400 <= status_code <= 499:
        return GoogleWorkspaceErrorKind.INVALID_REQUEST
    return GoogleWorkspaceErrorKind.MALFORMED_RESPONSE


def _parse_retry_after(
    headers: Mapping[str, str],
    *,
    policy: GoogleWorkspaceRetryPolicy,
) -> float | None:
    for key, value in headers.items():
        if key.casefold() != "retry-after":
            continue
        trimmed = value.strip()
        if not trimmed:
            return None
        if trimmed.isdigit():
            seconds = int(trimmed)
            if seconds < 0:
                return None
            return min(float(seconds), policy.max_retry_after_seconds)
        try:
            retry_at = parsedate_to_datetime(trimmed)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delta = (retry_at - datetime.now(timezone.utc)).total_seconds()
        if delta < 0:
            return None
        return min(delta, policy.max_retry_after_seconds)
    return None


def _decode_success_json(content: bytes, *, status_code: int, attempts: int) -> dict[str, object]:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="invalid_json",
            attempts=attempts,
        ) from None
    if not isinstance(payload, dict):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.MALFORMED_RESPONSE,
            status_code=status_code,
            retry_after_seconds=None,
            safe_reason="top_level_not_object",
            attempts=attempts,
        )
    return payload


def _validate_jitter_value(value: object, *, attempts: int) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.TEMPORARY,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_jitter_source",
            attempts=attempts,
        )
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.TEMPORARY,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_jitter_source",
            attempts=attempts,
        )
    return float(value)


def _compute_backoff_seconds(
    *,
    attempt_index: int,
    policy: GoogleWorkspaceRetryPolicy,
    jitter_source: _JitterSource,
    attempts: int,
) -> float:
    try:
        jitter = jitter_source()
    except Exception:
        raise GoogleWorkspaceApiError(
            kind=GoogleWorkspaceErrorKind.TEMPORARY,
            status_code=None,
            retry_after_seconds=None,
            safe_reason="invalid_jitter_source",
            attempts=attempts,
        ) from None
    validated_jitter = _validate_jitter_value(jitter, attempts=attempts)
    exponent = max(0, attempt_index - 1)
    raw = min(
        policy.base_backoff_seconds * (2**exponent),
        policy.max_backoff_seconds,
    )
    return min(raw * validated_jitter, policy.max_backoff_seconds)


def _api_error_from_response(
    *,
    status_code: int,
    headers: Mapping[str, str],
    content: bytes,
    attempts: int,
    policy: GoogleWorkspaceRetryPolicy,
) -> GoogleWorkspaceApiError:
    safe_reason = _extract_safe_reason(content)
    kind = _classify_status(status_code, safe_reason=safe_reason)
    retry_after = None
    if kind in _RETRYABLE_KINDS:
        retry_after = _parse_retry_after(headers, policy=policy)
    return GoogleWorkspaceApiError(
        kind=kind,
        status_code=status_code,
        retry_after_seconds=retry_after,
        safe_reason=safe_reason or kind.value,
        attempts=attempts,
    )


def parse_google_workspace_collection_page(
    payload: object,
    *,
    items_field: str,
) -> GoogleWorkspaceCollectionPage:
    validated_field = _validate_items_field(items_field)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")
    if validated_field not in payload:
        raise ValueError("missing items field")
    raw_items = payload[validated_field]
    if not isinstance(raw_items, list):
        raise ValueError("items field must be a list")
    items: list[dict[str, object]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("each item must be a dictionary")
        items.append(dict(item))
    next_token: GoogleWorkspacePageToken | None = None
    if "nextPageToken" in payload:
        raw_token = payload["nextPageToken"]
        if not isinstance(raw_token, str):
            raise ValueError("nextPageToken must be a string")
        next_token = GoogleWorkspacePageToken(value=raw_token)
    return GoogleWorkspaceCollectionPage(items=tuple(items), next_page_token=next_token)


def _execute_get_with_retry(
    transport: GoogleWorkspaceHttpTransport,
    *,
    url: str,
    query_params: dict[str, object] | None,
    request_headers: dict[str, str],
    success_max_bytes: int,
    binary_success: bool,
    decode_success: Callable[[int, dict[str, str], bytes, int], _T],
) -> _T:
    policy = transport._retry_policy
    last_error: GoogleWorkspaceApiError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            response = transport._executor.get(
                url=url,
                params=query_params,
                headers=request_headers,
                timeout_seconds=policy.request_timeout_seconds,
            )
        except GoogleWorkspaceApiError:
            raise
        except Exception:
            last_error = GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.TEMPORARY,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="executor_failure",
                attempts=attempt,
            )
        else:
            status_code, response_headers, content = _validate_response_shape(
                response,
                policy=policy,
                attempts=attempt,
                success_max_bytes=success_max_bytes,
                binary_success=binary_success,
            )
            if 200 <= status_code <= 299:
                return decode_success(status_code, response_headers, content, attempt)
            last_error = _api_error_from_response(
                status_code=status_code,
                headers=response_headers,
                content=content,
                attempts=attempt,
                policy=policy,
            )

        if last_error is None or not last_error.retryable or attempt >= policy.max_attempts:
            if last_error is not None:
                raise last_error
            raise GoogleWorkspaceApiError(
                kind=GoogleWorkspaceErrorKind.TEMPORARY,
                status_code=None,
                retry_after_seconds=None,
                safe_reason="unknown_failure",
                attempts=attempt,
            )

        delay = last_error.retry_after_seconds
        if delay is None:
            delay = _compute_backoff_seconds(
                attempt_index=attempt,
                policy=policy,
                jitter_source=transport._jitter_source,
                attempts=attempt,
            )
        transport._sleeper(delay)

    if last_error is not None:
        raise last_error
    raise GoogleWorkspaceApiError(
        kind=GoogleWorkspaceErrorKind.TEMPORARY,
        status_code=None,
        retry_after_seconds=None,
        safe_reason="unknown_failure",
        attempts=policy.max_attempts,
    )


class GoogleWorkspaceHttpTransport:
    """Bounded GET transport with retry ownership and safe provider error mapping."""

    def __init__(
        self,
        *,
        executor: GoogleWorkspaceRequestExecutor,
        retry_policy: GoogleWorkspaceRetryPolicy,
        sleeper: _Sleeper | None = None,
        jitter_source: _JitterSource | None = None,
    ) -> None:
        self._executor = executor
        self._retry_policy = retry_policy
        self._sleeper = sleeper or _default_sleeper
        self._jitter_source = jitter_source or _default_jitter

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        validated_path = _validate_relative_path(relative_path)
        query_params = _copy_params(params)
        request_headers = _copy_headers(headers)
        url = _build_service_url(source_kind=source_kind, relative_path=validated_path)
        policy = self._retry_policy

        def _decode_json(
            status_code: int,
            _headers: dict[str, str],
            content: bytes,
            attempts: int,
        ) -> dict[str, object]:
            return _decode_success_json(
                content,
                status_code=status_code,
                attempts=attempts,
            )

        return _execute_get_with_retry(
            self,
            url=url,
            query_params=query_params,
            request_headers=request_headers,
            success_max_bytes=policy.max_response_bytes,
            binary_success=False,
            decode_success=_decode_json,
        )

    def get_bytes(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None,
        expected_content_type: str,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        validated_path = _validate_relative_path(relative_path)
        query_params = _copy_params(params)
        validated_content_type = _validate_expected_content_type(expected_content_type)
        validated_max_bytes = _validate_binary_max_bytes(max_bytes)
        validated_range_mode = _validate_range_mode(range_limited)
        request_headers = _build_binary_request_headers(
            expected_content_type=validated_content_type,
            max_bytes=validated_max_bytes,
            range_limited=validated_range_mode,
        )
        url = _build_service_url(source_kind=source_kind, relative_path=validated_path)
        success_limit = (
            validated_max_bytes + 1 if validated_range_mode else validated_max_bytes
        )

        def _decode_binary(
            status_code: int,
            response_headers: dict[str, str],
            content: bytes,
            attempts: int,
        ) -> GoogleWorkspaceBinaryPayload:
            return _decode_success_binary(
                status_code=status_code,
                headers=response_headers,
                content=content,
                attempts=attempts,
                expected_content_type=validated_content_type,
                max_bytes=validated_max_bytes,
                range_limited=validated_range_mode,
            )

        return _execute_get_with_retry(
            self,
            url=url,
            query_params=query_params,
            request_headers=request_headers,
            success_max_bytes=success_limit,
            binary_success=True,
            decode_success=_decode_binary,
        )
