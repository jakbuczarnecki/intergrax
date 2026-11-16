# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MCP identity admission mode (MCP-RC MRC-1, OD-MCP-RC-01 D-4).

``HARNESS_BOUND`` (default) is fail-closed: a tool call is admitted only for a principal verified by the
canonical Harness Auth. Client-asserted tenant or user values never establish identity.

``LOCAL_DEV_EXPLICIT`` is a code-level opt-in (never an environment variable) that keeps the previous
trust-the-assertion behaviour for in-process calls and loopback clients only.
"""

from __future__ import annotations

import ipaddress
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.requests import Request


class McpIdentityMode(StrEnum):
    HARNESS_BOUND = "harness_bound"
    LOCAL_DEV_EXPLICIT = "local_dev_explicit"


DEFAULT_MCP_IDENTITY_MODE = McpIdentityMode.HARNESS_BOUND

_LOOPBACK_HOSTS = frozenset({"localhost", "testclient"})


def coerce_mcp_identity_mode(value: McpIdentityMode | str | None) -> McpIdentityMode:
    """Normalize an explicit composition value; ``None`` selects the fail-closed default."""
    if value is None:
        return DEFAULT_MCP_IDENTITY_MODE
    if isinstance(value, McpIdentityMode):
        return value
    try:
        return McpIdentityMode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown MCP identity mode: {value!r}") from exc


def is_loopback_client(request: Request | None) -> bool:
    """True for in-process calls (no HTTP request) and loopback HTTP clients."""
    if request is None:
        return True
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if not host:
        return False
    host = str(host).strip().lower()
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


__all__ = [
    "DEFAULT_MCP_IDENTITY_MODE",
    "McpIdentityMode",
    "coerce_mcp_identity_mode",
    "is_loopback_client",
]
