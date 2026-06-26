# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lazy MCP runtime imports for opt-in Tier-3 application surfaces.

This module intentionally has no FastMCP dependency. HTTP-only hosts import it only
inside ``if settings.include_mcp:`` to mount MCP with a clear error when optional
dependencies are missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from intergrax.utils import attribute_access

_F = TypeVar("_F", bound=Callable[..., Any])

_MCP_DEPENDENCY_MESSAGE = (
    "Tier-3 MCP surface is enabled (INCLUDE_MCP=true) but the FastMCP runtime is not "
    "available. Install project dependencies that provide fastmcp, or disable MCP with "
    "INCLUDE_MCP=false for HTTP-only startup."
)


class MCPDependencyError(ImportError):
    """Raised when MCP is enabled but FastMCP/MCP packages are unavailable."""


def ensure_mcp_dependencies() -> None:
    """Verify FastMCP is importable before loading MCP coupling code."""
    try:
        import fastmcp  # noqa: F401
    except ImportError as exc:
        raise MCPDependencyError(_MCP_DEPENDENCY_MESSAGE) from exc


def load_mcp_coupling() -> Callable[..., Any]:
    """Import :func:`couple_fastapi_with_mcp` after verifying MCP dependencies."""
    ensure_mcp_dependencies()
    from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp

    return couple_fastapi_with_mcp


def load_mcp_server_builder(import_path: str, *, symbol: str) -> Callable[..., Any]:
    """Import an application MCP server builder after verifying MCP dependencies."""
    ensure_mcp_dependencies()
    from importlib import import_module

    module = import_module(import_path)
    builder = attribute_access.optional(module, symbol, None)
    if builder is None:
        raise ImportError(f"MCP server builder {symbol!r} not found in {import_path!r}")
    return builder
