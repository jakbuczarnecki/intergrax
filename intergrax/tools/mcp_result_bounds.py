# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded MCP JSON surfaces — catalog description vs invocation result (transport layer only)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.tools.providers.provider_resource_enforcement import (
    DEFAULT_MCP_MAX_SERIALIZED_BYTES,
    bound_mcp_serialized_payload,
    utf8_byte_length,
)


def serialize_bounded_mcp_json(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MCP_MAX_SERIALIZED_BYTES,
) -> tuple[str, bool]:
    """Serialize catalog/description payloads with a hard UTF-8 byte ceiling (string output)."""
    payload = json.dumps(value, default=str, separators=(",", ":"))
    bounded, truncated = bound_mcp_serialized_payload(payload, max_bytes=max_bytes)
    return bounded, truncated


def bound_mcp_structured_result(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MCP_MAX_SERIALIZED_BYTES,
    overflow_error_code: str = "mcp_payload_byte_limit_exceeded",
) -> tuple[Any, bool]:
    """
    Bound logical MCP tool results (dict/list) without emitting invalid truncated JSON.

    When the serialized form exceeds ``max_bytes``, returns a typed error object instead.
    """
    payload = json.dumps(value, default=str, separators=(",", ":"))
    if utf8_byte_length(payload) <= max_bytes:
        return value, False
    return {"error": overflow_error_code}, True


__all__ = [
    "DEFAULT_MCP_MAX_SERIALIZED_BYTES",
    "bound_mcp_structured_result",
    "serialize_bounded_mcp_json",
]
