# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded MCP catalog serialization (transport layer — not lifecycle authority)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.tools.providers.provider_resource_enforcement import (
    DEFAULT_MCP_MAX_SERIALIZED_BYTES,
    bound_mcp_serialized_payload,
)


def serialize_bounded_mcp_json(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MCP_MAX_SERIALIZED_BYTES,
) -> tuple[str, bool]:
    """Serialize tool/MCP payloads with a hard UTF-8 byte ceiling."""
    payload = json.dumps(value, default=str, separators=(",", ":"))
    bounded, truncated = bound_mcp_serialized_payload(payload, max_bytes=max_bytes)
    return bounded, truncated


__all__ = ["DEFAULT_MCP_MAX_SERIALIZED_BYTES", "serialize_bounded_mcp_json"]
