# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed provider-local resource ceilings (TOOL-LIFE-X Batch D — adapter layer only)."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_DATABASE_MAX_ROWS = 1_000
DEFAULT_DATABASE_MAX_RESULT_BYTES = 4_194_304
DEFAULT_HTTP_MAX_RESPONSE_BYTES = 4_194_304
DEFAULT_BROWSER_MAX_TEXT_CHARS = 200_000
DEFAULT_MCP_MAX_SERIALIZED_BYTES = 1_048_576


class ProviderResourceLimitExceeded(RuntimeError):
    """Physical result exceeds a declared hard ceiling (not a transport timeout)."""

    def __init__(self, code: str, *, detail: str = "") -> None:
        self.code = code
        super().__init__(detail or code)


def utf8_byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def truncate_utf8_text(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


def truncate_utf8_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def estimate_rows_json_bytes(rows: list[dict[str, Any]]) -> int:
    return utf8_byte_length(json.dumps(rows, default=str, separators=(",", ":")))


def bound_database_query_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    max_result_bytes: int,
) -> tuple[list[dict[str, Any]], bool, str]:
    """
    Apply row and UTF-8 JSON byte ceilings on fetched SQL results.

    Returns ``(rows, truncated, limit_reason)`` where ``limit_reason`` is
    ``row_limit``, ``byte_limit``, or empty when not truncated.
    """
    if max_rows < 1 or max_result_bytes < 1:
        raise ProviderResourceLimitExceeded("invalid_bounds")

    working = list(rows)
    truncated = False
    reason = ""

    if len(working) > max_rows:
        working = working[:max_rows]
        truncated = True
        reason = "row_limit"

    while working and estimate_rows_json_bytes(working) > max_result_bytes:
        if len(working) == 1:
            raise ProviderResourceLimitExceeded(
                "database_result_byte_limit",
                detail="single_row_exceeds_max_result_bytes",
            )
        working = working[: max(1, len(working) // 2)]
        truncated = True
        reason = "byte_limit"

    return working, truncated, reason


def bound_http_response_body(body: str, *, max_response_bytes: int) -> tuple[str, bool]:
    if max_response_bytes < 1:
        raise ProviderResourceLimitExceeded("invalid_bounds")
    return truncate_utf8_text(body, max_response_bytes)


def bound_mcp_serialized_payload(payload: str, *, max_bytes: int) -> tuple[str, bool]:
    if max_bytes < 1:
        raise ProviderResourceLimitExceeded("invalid_bounds")
    return truncate_utf8_text(payload, max_bytes)


__all__ = [
    "DEFAULT_BROWSER_MAX_TEXT_CHARS",
    "DEFAULT_DATABASE_MAX_RESULT_BYTES",
    "DEFAULT_DATABASE_MAX_ROWS",
    "DEFAULT_HTTP_MAX_RESPONSE_BYTES",
    "DEFAULT_MCP_MAX_SERIALIZED_BYTES",
    "ProviderResourceLimitExceeded",
    "bound_database_query_rows",
    "bound_http_response_body",
    "bound_mcp_serialized_payload",
    "estimate_rows_json_bytes",
    "truncate_utf8_chars",
    "truncate_utf8_text",
    "utf8_byte_length",
]
